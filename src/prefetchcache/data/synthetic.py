"""Synthetic clickstream generator.

Used for unit tests and for exercising the pipeline before the real datasets
are available. Numbers produced from this data are NOT research results.
"""
from __future__ import annotations

import numpy as np

from .sessions import RawSession


def make_synthetic(
    n_users: int = 2000,
    n_pages: int = 46,
    mean_sessions: float = 3.0,
    mean_length: float = 8.0,
    concentration: float = 0.15,
    with_times: bool = False,
    seed: int = 0,
) -> list[RawSession]:
    """Sessions from a random first-order Markov chain over ``n_pages`` pages.

    A small ``concentration`` makes transitions peaky (predictable); a large
    one makes them close to uniform.
    """
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.full(n_pages, concentration), size=n_pages)
    start = rng.dirichlet(np.full(n_pages, 0.5))
    cum_trans = np.cumsum(trans, axis=1)
    cum_start = np.cumsum(start)
    sessions: list[RawSession] = []
    clock = 0.0
    for u in range(n_users):
        for k in range(1 + rng.poisson(mean_sessions - 1)):
            length = 2 + rng.geometric(1.0 / max(mean_length - 2, 1.0)) - 1
            page = int(np.searchsorted(cum_start, rng.random()))
            pages = [min(page, n_pages - 1)]
            for _ in range(length - 1):
                nxt = int(np.searchsorted(cum_trans[pages[-1]], rng.random()))
                pages.append(min(nxt, n_pages - 1))
            times = None
            if with_times:
                clock += rng.exponential(5.0)
                gaps = rng.exponential(20.0, size=length)
                times = tuple((clock + np.cumsum(gaps)).tolist())
            sessions.append(RawSession(f"u{u}", f"u{u}-s{k}", tuple(f"GET /pages/{p}" for p in pages), times))
    return sessions
