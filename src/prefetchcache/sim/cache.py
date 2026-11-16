"""Simulated application cache: per-session keys, TTL, optional LRU byte capacity.

Mirrors what the Redis layer will do in the middleware. Keys are
(session, endpoint), so an entry can never be served to another session (NFR3).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

Key = tuple[int, int]  # (session index, endpoint id)


@dataclass
class Entry:
    size: int
    expires_at: float
    prefetched: bool
    used: bool = False


class SessionCache:
    def __init__(self, capacity_bytes: int | None = None, on_remove: Callable[[Key, Entry, str], None] | None = None):
        self.capacity = capacity_bytes
        self.bytes_used = 0
        self._entries: OrderedDict[Key, Entry] = OrderedDict()
        self._by_session: dict[int, set[int]] = {}
        self._on_remove = on_remove

    def __len__(self) -> int:
        return len(self._entries)

    def _remove(self, key: Key, reason: str) -> None:
        entry = self._entries.pop(key)
        self.bytes_used -= entry.size
        eps = self._by_session[key[0]]
        eps.discard(key[1])
        if not eps:
            del self._by_session[key[0]]
        if self._on_remove:
            self._on_remove(key, entry, reason)

    def get(self, key: Key, now: float) -> Entry | None:
        """Return a fresh entry and mark it most recently used, else None."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if now >= entry.expires_at:
            self._remove(key, "expired")
            return None
        self._entries.move_to_end(key)
        return entry

    def put(self, key: Key, size: int, expires_at: float, prefetched: bool) -> bool:
        if self.capacity is not None and size > self.capacity:
            return False
        if key in self._entries:
            self._remove(key, "replaced")
        while self.capacity is not None and self.bytes_used + size > self.capacity:
            self._remove(next(iter(self._entries)), "evicted")
        self._entries[key] = Entry(size, expires_at, prefetched)
        self._by_session.setdefault(key[0], set()).add(key[1])
        self.bytes_used += size
        return True

    def drop_session(self, session: int) -> None:
        for ep in list(self._by_session.get(session, ())):
            self._remove((session, ep), "session_end")

    def clear(self) -> None:
        for key in list(self._entries):
            self._remove(key, "flush")
