"""Token-bucket byte budget (FR3)."""
from __future__ import annotations


class TokenBucket:
    """Bytes refill at ``rate`` per time unit up to ``capacity``.

    The time unit is whatever clock the caller uses: seconds for the live
    middleware and Retailrocket replay, requests for the coffee-shop replay.
    Invariant: bytes granted by time t never exceed capacity + rate * (t - t0).
    """

    def __init__(self, rate: float, capacity: float, start_time: float = 0.0, start_full: bool = True):
        if rate < 0 or capacity <= 0:
            raise ValueError("rate must be >= 0 and capacity > 0")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.tokens = self.capacity if start_full else 0.0
        self._last = float(start_time)
        self.granted_bytes = 0

    def _refill(self, now: float) -> None:
        if now > self._last:
            self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
            self._last = now

    def available(self, now: float) -> float:
        self._refill(now)
        return self.tokens

    def try_consume(self, n_bytes: int, now: float) -> bool:
        self._refill(now)
        if n_bytes > self.tokens:
            return False
        self.tokens -= n_bytes
        self.granted_bytes += n_bytes
        return True

    def charge(self, n_bytes: int, now: float) -> None:
        """Charge bytes that were already transferred, going into debt if needed.

        Used when a prefetched response turns out larger than its known size;
        the debt delays later prefetches, so the long-run rate still holds.
        """
        self._refill(now)
        self.tokens -= n_bytes
        self.granted_bytes += n_bytes
