"""Budget sweep: replay the same sessions at several byte budgets.

Each budget is a fraction of the bytes that ungated top-k prefetching spends
with the same predictor and no limit, converted to a token-bucket rate.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.sessions import Session
from ..policy.budget import TokenBucket
from ..policy.gate import PrefetchPolicy
from ..predictors.base import Predictor
from .metrics import Metrics
from .replay import SimConfig, event_stream, replay, replay_duration

# Proposed levels, to be confirmed with the supervisor.
DEFAULT_BUDGET_FRACTIONS = (1.0, 0.5, 0.25, 0.10, 0.05)


def reference_bytes(sessions, predictor, sizes, cacheable, top_k: int, config: SimConfig) -> int:
    """Bytes prefetched by ungated top-k with no budget."""
    return replay(sessions, predictor, sizes, cacheable, PrefetchPolicy(top_k, None), config).prefetched_bytes


def make_bucket(fraction: float, ref_bytes: int, sessions: Sequence[Session], sizes: np.ndarray, burst_units: float) -> TokenBucket:
    duration = max(replay_duration(sessions), 1.0)
    rate = fraction * ref_bytes / duration
    # The bucket must be able to hold the largest response, or it could never be prefetched.
    capacity = max(rate * burst_units, float(sizes.max()))
    start = next(event_stream(sessions))[0]
    return TokenBucket(rate, capacity, start_time=start, start_full=False)


def budget_sweep(
    sessions: Sequence[Session],
    predictor: Predictor,
    sizes: np.ndarray,
    cacheable: Sequence[bool],
    policy: PrefetchPolicy,
    config: SimConfig,
    fractions: Sequence[float] = DEFAULT_BUDGET_FRACTIONS,
    burst_units: float = 20.0,
) -> dict[float, Metrics]:
    ref = reference_bytes(sessions, predictor, sizes, cacheable, policy.top_k, config)
    out: dict[float, Metrics] = {}
    for f in fractions:
        bucket = make_bucket(f, ref, sessions, sizes, burst_units)
        out[f] = replay(sessions, predictor, sizes, cacheable, policy, config, bucket)
    return out
