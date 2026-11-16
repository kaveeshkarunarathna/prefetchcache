"""Prefetch engine: the request flow of deck slide 8, independent of the web framework.

1. Serve the user's request from the per-session cache or the backend.
2. After the response is ready, ask the predictor for likely next endpoints.
3. Apply the gate (fixed threshold or budget-linked) and the token bucket.
4. Prefetch into the cache with a TTL.
5. Log every prefetch as used or wasted, with its size in bytes.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from ..policy.budget import TokenBucket
from ..policy.gate import PrefetchPolicy
from ..policy.linked import BudgetLinkedPolicy
from ..sim.metrics import Metrics
from .store import CacheStore, cache_key

Fetcher = Callable[[str], Awaitable[bytes]]  # endpoint path -> response body


@dataclass
class _Pending:
    """A prefetched response that has not been used yet."""

    session_id: str
    endpoint: str
    size: int
    expires_at: float


class PrefetchEngine:
    def __init__(
        self,
        store: CacheStore,
        fetcher: Fetcher,
        predictors: dict[str, object],
        vocab: Sequence[str],
        model: str,
        policy: PrefetchPolicy | BudgetLinkedPolicy,
        bucket: TokenBucket | None,
        ttl_seconds: float = 300.0,
        history_length: int = 20,
        max_sessions: int = 100_000,
        log_size: int = 1_000,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.store = store
        self.fetcher = fetcher
        self.predictors = predictors
        self.vocab = list(vocab)
        self._index = {name: i for i, name in enumerate(self.vocab)}
        self.model = model
        self.policy = policy
        self.bucket = bucket
        self.ttl = ttl_seconds
        self.history_length = history_length
        self.max_sessions = max_sessions
        self.clock = clock
        self.metrics = Metrics()
        self.errors = 0
        self.log: deque[dict] = deque(maxlen=log_size)
        self._history: OrderedDict[str, deque[int]] = OrderedDict()
        self._sizes: dict[str, int] = {}  # response size per endpoint, learned from real responses
        self._pending: dict[str, _Pending] = {}
        self._in_flight: set[str] = set()  # keys being prefetched right now
        self._tasks: set[asyncio.Task] = set()

    # ---- serving -------------------------------------------------------

    async def lookup(self, session_id: str, endpoint: str) -> bytes | None:
        """Return the cached response for this session, or None on a miss."""
        self._expire()
        key = cache_key(session_id, endpoint)
        body = await self.store.get(key)
        self.metrics.requests += 1
        self.metrics.cacheable_requests += 1
        if body is None:
            return None
        self.metrics.hits += 1
        pending = self._pending.pop(key, None)
        if pending is not None:
            self.metrics.useful_prefetches += 1
            self.metrics.useful_bytes += pending.size
            self._record(pending, "used")
        return body

    async def store_response(self, session_id: str, endpoint: str, body: bytes) -> None:
        """Cache a response the backend just produced for this session."""
        self._sizes[endpoint] = len(body)
        self.metrics.backend_bytes += len(body)
        await self.store.set(cache_key(session_id, endpoint), body, self.ttl)

    def after_response(self, session_id: str, endpoint: str) -> None:
        """Record the request and start prefetching in the background.

        Called once the user's response is ready, so prefetching can never
        delay it (NFR2).
        """
        history = self._history.get(session_id)
        if history is None:
            history = self._history[session_id] = deque(maxlen=self.history_length)
            if len(self._history) > self.max_sessions:
                self._history.popitem(last=False)
        else:
            self._history.move_to_end(session_id)
        history.append(self._index.get(f"GET {endpoint}", 0))
        task = asyncio.ensure_future(self._prefetch_safely(session_id, tuple(history)))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for running prefetches (tests and shutdown)."""
        while self._tasks:
            running = [t for t in self._tasks if not t.done()]
            if running:
                await asyncio.wait(running)
            else:
                await asyncio.sleep(0)  # let finished tasks remove themselves

    # ---- prefetching ---------------------------------------------------

    async def _prefetch_safely(self, session_id: str, history: tuple[int, ...]) -> None:
        try:
            await self._prefetch(session_id, history)
        except Exception:  # a prefetch failure must never reach the user (NFR2)
            self.errors += 1

    async def _prefetch(self, session_id: str, history: tuple[int, ...]) -> None:
        policy = self.policy
        if policy.top_k <= 0:
            return
        predictor = self.predictors[self.model]
        started = time.perf_counter()
        ranked = predictor.predict(history, 10)
        self.metrics.decision_ms.append((time.perf_counter() - started) * 1000)
        if len(self.metrics.decision_ms) > 10_000:
            del self.metrics.decision_ms[:5_000]
        now = self.clock()
        linked = isinstance(policy, BudgetLinkedPolicy)
        if linked:
            policy.observe(self.bucket.available(now) / self.bucket.capacity if self.bucket else 1.0)
        chosen = 0
        for candidate, probability in ranked:
            if chosen >= policy.top_k:
                break
            name = self.vocab[candidate]
            if not name.startswith("GET "):
                continue  # only safe GET endpoints are prefetched
            endpoint = name[4:]
            size = self._sizes.get(endpoint)
            if size is None:
                continue  # size unknown: cannot be charged to the budget yet
            if linked:
                if not policy.admit(probability, size):
                    self.metrics.gate_skipped += 1
                    continue
            elif not policy.passes(probability):
                self.metrics.gate_skipped += 1
                break
            chosen += 1
            key = cache_key(session_id, endpoint)
            if key in self._in_flight or key in self._pending or await self.store.get(key) is not None:
                self.metrics.already_cached += 1
                continue
            if self.bucket is not None and not self.bucket.try_consume(size, now):
                self.metrics.budget_skipped += 1
                continue
            self._in_flight.add(key)
            try:
                body = await self.fetcher(endpoint)
                actual = len(body)
                if actual > size and self.bucket is not None:
                    self.bucket.charge(actual - size, now)  # response grew since it was last seen
                self._sizes[endpoint] = actual
                await self.store.set(key, body, self.ttl)
            except Exception:
                if self.bucket is not None:
                    self.bucket.charge(-size, now)  # nothing was cached: refund
                raise
            finally:
                self._in_flight.discard(key)
            self.metrics.prefetches += 1
            self.metrics.prefetched_bytes += actual
            self._pending[key] = _Pending(session_id, endpoint, actual, now + self.ttl)

    # ---- accounting ----------------------------------------------------

    def _record(self, pending: _Pending, outcome: str) -> None:
        self.log.append({"session": pending.session_id, "endpoint": pending.endpoint, "bytes": pending.size, "outcome": outcome})

    def _expire(self) -> None:
        """Count prefetched responses whose TTL passed without a request as wasted."""
        now = self.clock()
        for key in [k for k, p in self._pending.items() if now >= p.expires_at]:
            pending = self._pending.pop(key)
            self.metrics.wasted_prefetches += 1
            self.metrics.wasted_bytes += pending.size
            self._record(pending, "wasted")

    def reset_stats(self) -> None:
        """Zero the counters and the log; cached data and configuration stay."""
        self.metrics = Metrics()
        self.errors = 0
        self.log.clear()
        self._pending.clear()

    def stats(self) -> dict:
        self._expire()
        out = self.metrics.summary()
        out["pending_prefetches"] = len(self._pending)
        out["pending_bytes"] = sum(p.size for p in self._pending.values())
        out["prefetch_errors"] = self.errors
        out["budget_tokens"] = self.bucket.available(self.clock()) if self.bucket else None
        return out

    # ---- runtime configuration (FR4) -----------------------------------

    def config(self) -> dict:
        policy = self.policy
        return {
            "model": self.model,
            "models": sorted(self.predictors),
            "policy": "linked" if isinstance(policy, BudgetLinkedPolicy) else "fixed",
            "top_k": policy.top_k,
            "threshold": getattr(policy, "threshold", None),
            "min_confidence": getattr(policy, "min_confidence", None),
            "budget_bytes_per_second": self.bucket.rate if self.bucket else None,
            "budget_burst_bytes": self.bucket.capacity if self.bucket else None,
            "ttl_seconds": self.ttl,
        }

    def configure(
        self,
        model: str | None = None,
        policy: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        min_confidence: float | None = None,
        budget_bytes_per_second: float | None = None,
        budget_burst_bytes: float | None = None,
        ttl_seconds: float | None = None,
    ) -> dict:
        if model is not None:
            if model not in self.predictors:
                raise ValueError(f"unknown model {model!r}; available: {sorted(self.predictors)}")
            self.model = model
        current = self.config()
        kind = policy or current["policy"]
        if kind not in ("fixed", "linked"):
            raise ValueError("policy must be 'fixed' or 'linked'")
        k = current["top_k"] if top_k is None else top_k
        if k < 0:
            raise ValueError("top_k must be 0 or more")
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if kind == "linked":
            floor = min_confidence if min_confidence is not None else (current["min_confidence"] or 0.0)
            if not isinstance(self.policy, BudgetLinkedPolicy) or top_k is not None or min_confidence is not None:
                price = self.policy.price if isinstance(self.policy, BudgetLinkedPolicy) else 0.0
                self.policy = BudgetLinkedPolicy(k, floor, price=price)
        else:
            t = threshold if threshold is not None else current["threshold"]
            self.policy = PrefetchPolicy(k, t)
        if budget_bytes_per_second is not None or budget_burst_bytes is not None:
            rate = budget_bytes_per_second if budget_bytes_per_second is not None else (self.bucket.rate if self.bucket else 0.0)
            burst = budget_burst_bytes if budget_burst_bytes is not None else (self.bucket.capacity if self.bucket else None)
            if burst is None:
                raise ValueError("budget_burst_bytes is required when no budget is set")
            now = self.clock()
            tokens = min(self.bucket.available(now), burst) if self.bucket else 0.0
            self.bucket = TokenBucket(rate, burst, start_time=now, start_full=False)
            self.bucket.tokens = tokens
        if ttl_seconds is not None:
            if ttl_seconds <= 0:
                raise ValueError("ttl_seconds must be positive")
            self.ttl = ttl_seconds
        return self.config()
