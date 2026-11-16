"""Trace-driven replay of sessions through cache, predictor, gate and byte budget.

Follows the request flow on deck slide 8: serve the request from the cache or
the backend, then ask the predictor, apply the confidence gate, check the
token bucket, prefetch, and record each prefetch as used or wasted with its
size in bytes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from ..data.sessions import Session
from ..policy.budget import TokenBucket
from ..policy.gate import PrefetchPolicy
from ..predictors.base import Predictor
from .cache import Entry, Key, SessionCache
from .metrics import Metrics


@dataclass(frozen=True)
class SimConfig:
    """``ttl`` is in clock units: seconds with real timestamps, requests on the logical clock."""

    ttl: float
    cache_capacity_bytes: int | None = None
    top_n_candidates: int = 10
    time_decisions: bool = False


def uses_timestamps(sessions: Sequence[Session]) -> bool:
    return bool(sessions) and all(s.times is not None for s in sessions)


def event_stream(sessions: Sequence[Session]) -> Iterator[tuple[float, int, int]]:
    """Yield (time, session index, position) in replay order.

    With timestamps, all sessions are merged by time. Without them (coffee
    data), sessions are replayed one after another on a logical clock that
    advances by one per request.
    """
    if uses_timestamps(sessions):
        events = [(t, si, pos) for si, s in enumerate(sessions) for pos, t in enumerate(s.times)]
        events.sort()
        yield from events
    else:
        tick = 0
        for si, s in enumerate(sessions):
            for pos in range(len(s.endpoints)):
                yield float(tick), si, pos
                tick += 1


def replay_duration(sessions: Sequence[Session]) -> float:
    if uses_timestamps(sessions):
        return max(s.times[-1] for s in sessions) - min(s.times[0] for s in sessions)
    return float(sum(len(s.endpoints) for s in sessions))


def replay(
    sessions: Sequence[Session],
    predictor: Predictor | None,
    sizes: np.ndarray,
    cacheable: Sequence[bool],
    policy,
    config: SimConfig,
    bucket: TokenBucket | None = None,
    prefetch_log: list[dict] | None = None,
) -> Metrics:
    m = Metrics()

    def on_remove(key: Key, entry: Entry, reason: str) -> None:
        if entry.prefetched and not entry.used:
            m.wasted_prefetches += 1
            m.wasted_bytes += entry.size
            if prefetch_log is not None:
                prefetch_log.append({"session": key[0], "endpoint": key[1], "bytes": entry.size, "outcome": "wasted", "reason": reason})

    cache = SessionCache(config.cache_capacity_bytes, on_remove)
    do_prefetch = predictor is not None and policy.top_k > 0
    linked = hasattr(policy, "admit")  # budget-linked gate: decision depends on size and bucket fill

    for now, si, pos in event_stream(sessions):
        session = sessions[si]
        ep = session.endpoints[pos]
        m.requests += 1

        # 1. Serve the user's request first.
        if cacheable[ep]:
            m.cacheable_requests += 1
            entry = cache.get((si, ep), now)
            if entry is not None:
                m.hits += 1
                if entry.prefetched and not entry.used:
                    entry.used = True
                    m.useful_prefetches += 1
                    m.useful_bytes += entry.size
                    if prefetch_log is not None:
                        prefetch_log.append({"session": si, "endpoint": ep, "bytes": entry.size, "outcome": "used", "reason": "hit"})
            else:
                size = int(sizes[ep])
                m.backend_bytes += size
                cache.put((si, ep), size, now + config.ttl, prefetched=False)
        else:
            m.backend_bytes += int(sizes[ep])

        # 2. Decide what to prefetch. The server cannot know a session has
        #    ended, so this also runs after the session's last request.
        if do_prefetch:
            history = session.endpoints[: pos + 1]
            if config.time_decisions:
                t0 = time.perf_counter()
                ranked = predictor.predict(history, config.top_n_candidates)
                m.decision_ms.append((time.perf_counter() - t0) * 1000)
            else:
                ranked = predictor.predict(history, config.top_n_candidates)
            if linked:
                policy.observe(bucket.available(now) / bucket.capacity if bucket is not None else 1.0)
            chosen = 0
            for cand, prob in ranked:
                if chosen >= policy.top_k:
                    break
                if not cacheable[cand]:
                    continue
                size = int(sizes[cand])
                if linked:
                    if not policy.admit(prob, size):
                        m.gate_skipped += 1
                        continue  # a smaller, less likely candidate may still be worth its bytes
                elif not policy.passes(prob):
                    m.gate_skipped += 1
                    break  # ranked by probability, so the rest fail too
                chosen += 1
                if cache.get((si, cand), now) is not None:
                    m.already_cached += 1
                    continue
                if bucket is not None and not bucket.try_consume(size, now):
                    m.budget_skipped += 1
                    continue
                if cache.put((si, cand), size, now + config.ttl, prefetched=True):
                    m.prefetches += 1
                    m.prefetched_bytes += size

        if pos == len(session.endpoints) - 1:
            cache.drop_session(si)

    cache.clear()
    return m
