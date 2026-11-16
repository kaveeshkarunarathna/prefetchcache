"""Cache stores for the middleware. Keys are always scoped to one session (NFR3)."""
from __future__ import annotations

import time
from typing import Callable, Protocol


def cache_key(session_id: str, endpoint: str) -> str:
    """Per-session key, so a cached response can never be served to another session."""
    return f"pc:{len(session_id)}:{session_id}:{endpoint}"


class CacheStore(Protocol):
    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl_seconds: float) -> None: ...

    async def delete(self, key: str) -> None: ...


class MemoryStore:
    """In-process store with TTL, for tests and single-process runs."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._data: dict[str, tuple[bytes, float]] = {}

    async def get(self, key: str) -> bytes | None:
        item = self._data.get(key)
        if item is None:
            return None
        if self._clock() >= item[1]:
            del self._data[key]
            return None
        return item[0]

    async def set(self, key: str, value: bytes, ttl_seconds: float) -> None:
        self._data[key] = (value, self._clock() + ttl_seconds)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


class RedisStore:
    """Redis-backed store; expiry is handled by Redis per key."""

    def __init__(self, client):
        self._redis = client  # redis.asyncio.Redis (or a compatible client)

    async def get(self, key: str) -> bytes | None:
        return await self._redis.get(key)

    async def set(self, key: str, value: bytes, ttl_seconds: float) -> None:
        await self._redis.set(key, value, px=max(1, int(ttl_seconds * 1000)))

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)
