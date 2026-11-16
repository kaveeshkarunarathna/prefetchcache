"""Common session representation shared by every dataset loader."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

UNK = "<unk>"


@dataclass(frozen=True)
class RawSession:
    """One user session as loaded from a dataset, before vocabulary encoding.

    ``times`` is None when the dataset has no per-request timestamps
    (the coffee-shop data); the simulator then uses a logical clock.
    """

    user_id: str
    session_id: str
    endpoints: tuple[str, ...]
    times: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.times is not None and len(self.times) != len(self.endpoints):
            raise ValueError("times and endpoints must have the same length")


@dataclass(frozen=True)
class Session:
    """A session with endpoints encoded as vocabulary ids."""

    user_id: str
    session_id: str
    endpoints: tuple[int, ...]
    times: tuple[float, ...] | None = None


def is_cacheable(endpoint: str) -> bool:
    """Only safe GET endpoints are cached or prefetched (deck slide 15)."""
    return endpoint.startswith("GET ")


class Vocab:
    """Endpoint vocabulary, fitted on training sessions only.

    Id 0 is reserved for unknown endpoints, which are never prefetched.
    """

    def __init__(self, names: Sequence[str]):
        if not names or names[0] != UNK:
            raise ValueError("vocabulary must start with the unknown token")
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.cacheable = [is_cacheable(n) for n in self.names]

    @classmethod
    def fit(cls, sessions: Iterable[RawSession], min_count: int = 1) -> "Vocab":
        counts: Counter[str] = Counter()
        for s in sessions:
            counts.update(s.endpoints)
        kept = sorted(n for n, c in counts.items() if c >= min_count)
        return cls([UNK, *kept])

    def __len__(self) -> int:
        return len(self.names)

    def encode(self, sessions: Iterable[RawSession]) -> list[Session]:
        idx = self.index
        return [
            Session(s.user_id, s.session_id, tuple(idx.get(e, 0) for e in s.endpoints), s.times)
            for s in sessions
        ]
