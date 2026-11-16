import numpy as np

from prefetchcache.policy.budget import TokenBucket
from prefetchcache.policy.gate import PrefetchPolicy


def test_gate_threshold():
    assert PrefetchPolicy(1, None).passes(0.01)
    gated = PrefetchPolicy(1, 0.5)
    assert gated.passes(0.5) and gated.passes(0.9) and not gated.passes(0.49)


def test_bucket_refills_and_caps():
    b = TokenBucket(rate=10, capacity=100, start_full=False)
    assert not b.try_consume(1, now=0)
    assert b.try_consume(50, now=5)
    assert not b.try_consume(1, now=5)
    assert b.available(now=1000) == 100  # capped at capacity


def test_bucket_never_exceeds_budget():
    rng = np.random.default_rng(0)
    rate, cap = 500.0, 4000.0
    b = TokenBucket(rate, cap, start_time=0.0)
    now = 0.0
    for _ in range(5000):
        now += rng.exponential(0.5)
        b.try_consume(int(rng.integers(1, 3000)), now)
        assert b.granted_bytes <= cap + rate * now + 1e-6
