import numpy as np

from prefetchcache.data.sessions import Session, Vocab
from prefetchcache.data.sizes import lognormal_sizes
from prefetchcache.data.split import split_sessions
from prefetchcache.data.synthetic import make_synthetic
from prefetchcache.policy.budget import TokenBucket
from prefetchcache.policy.gate import PrefetchPolicy
from prefetchcache.policy.linked import BudgetLinkedPolicy
from prefetchcache.predictors.baselines import MarkovPredictor, PopularityPredictor
from prefetchcache.sim.cache import SessionCache
from prefetchcache.sim.metrics import mean_ci95
from prefetchcache.sim.replay import SimConfig, replay
from prefetchcache.sim.sweep import budget_sweep, reference_bytes


class Fixed:
    """Predicts a fixed next endpoint per current endpoint."""

    name = "fixed"

    def __init__(self, table):
        self.table = table

    def predict(self, history, top_n=5):
        return self.table.get(history[-1], [])[:top_n]


SIZES = np.array([0, 100, 200, 400, 800])
CACHEABLE = [False, True, True, True, True]


def test_used_and_wasted_accounting():
    # 1 -> predict 2 (correct, used); 2 -> predict 4 (wrong, wasted); 3 -> nothing.
    pred = Fixed({1: [(2, 0.9)], 2: [(4, 0.6)]})
    log = []
    m = replay([Session("u", "s", (1, 2, 3))], pred, SIZES, CACHEABLE, PrefetchPolicy(1), SimConfig(ttl=10), prefetch_log=log)
    assert (m.prefetches, m.useful_prefetches, m.wasted_prefetches) == (2, 1, 1)
    assert (m.prefetched_bytes, m.useful_bytes, m.wasted_bytes) == (1000, 200, 800)
    assert m.hits == 1 and m.cacheable_requests == 3
    assert m.backend_bytes == 100 + 400  # request for 2 was served from the prefetch
    assert m.precision == 0.5 and m.useful_per_mb == 1 / 0.001
    # FR5: the log agrees with the counters.
    assert sum(r["bytes"] for r in log if r["outcome"] == "used") == m.useful_bytes
    assert sum(r["bytes"] for r in log if r["outcome"] == "wasted") == m.wasted_bytes


def test_gate_blocks_low_confidence():
    pred = Fixed({1: [(2, 0.9)], 2: [(4, 0.6)]})
    m = replay([Session("u", "s", (1, 2, 3))], pred, SIZES, CACHEABLE, PrefetchPolicy(1, 0.7), SimConfig(ttl=10))
    assert (m.prefetches, m.useful_prefetches, m.wasted_bytes, m.gate_skipped) == (1, 1, 0, 1)


def test_prefetch_expires_after_ttl():
    pred = Fixed({1: [(2, 0.9)]})
    sessions = [Session("u", "s", (1, 3, 3, 2))]
    assert replay(sessions, pred, SIZES, CACHEABLE, PrefetchPolicy(1), SimConfig(ttl=3)).useful_prefetches == 0
    assert replay(sessions, pred, SIZES, CACHEABLE, PrefetchPolicy(1), SimConfig(ttl=4)).useful_prefetches == 1


def test_no_cross_session_serving():
    # NFR3: session A's prefetch of endpoint 2 must not serve session B.
    pred = Fixed({1: [(2, 0.9)]})
    sessions = [Session("a", "a", (1, 3), (0.0, 10.0)), Session("b", "b", (2, 3), (1.0, 2.0))]
    m = replay(sessions, pred, SIZES, CACHEABLE, PrefetchPolicy(1), SimConfig(ttl=100))
    assert m.hits == 0 and m.useful_prefetches == 0 and m.wasted_bytes == 200


def test_post_endpoints_never_cached_or_prefetched():
    pred = Fixed({1: [(0, 0.9), (2, 0.1)]})
    m = replay([Session("u", "s", (1, 0, 0))], pred, SIZES, CACHEABLE, PrefetchPolicy(1), SimConfig(ttl=10))
    assert m.cacheable_requests == 1 and m.hits == 0
    assert m.prefetched_bytes == 200  # skipped the non-cacheable candidate, took endpoint 2


def test_lru_capacity_eviction():
    removed = []
    cache = SessionCache(capacity_bytes=300, on_remove=lambda k, e, r: removed.append((k, r)))
    cache.put((0, 1), 100, 99, False)
    cache.put((0, 2), 100, 99, False)
    cache.get((0, 1), 0)  # touch 1, so 2 is least recently used
    cache.put((0, 3), 200, 99, False)
    assert removed == [((0, 2), "evicted")] and cache.bytes_used == 300


def _fitted():
    split = split_sessions(make_synthetic(n_users=400, seed=3), seed=0)
    vocab = Vocab.fit(split.train)
    train, val = vocab.encode(split.train), vocab.encode(split.val)
    return vocab, train, val


def test_bytes_balance_and_budget_respected():
    vocab, train, val = _fitted()
    sizes = lognormal_sizes(len(vocab), seed=0)
    pred = MarkovPredictor().fit(train, len(vocab))
    config = SimConfig(ttl=5)
    ref = reference_bytes(val, pred, sizes, vocab.cacheable, 1, config)
    sweep = budget_sweep(val, pred, sizes, vocab.cacheable, PrefetchPolicy(1, None), config)
    n_requests = sum(len(s.endpoints) for s in val)
    for fraction, m in sweep.items():
        assert m.prefetched_bytes == m.useful_bytes + m.wasted_bytes
        assert m.prefetches == m.useful_prefetches + m.wasted_prefetches
        # FR3: bytes never exceed the budget (rate * duration + one burst).
        rate = fraction * ref / n_requests
        assert m.prefetched_bytes <= rate * n_requests + max(rate * 20, sizes.max()) + 1e-6
    assert sweep[0.05].prefetched_bytes < sweep[0.5].prefetched_bytes <= ref


def test_markov_beats_popularity_and_no_prefetch():
    vocab, train, val = _fitted()
    sizes = lognormal_sizes(len(vocab), seed=0)
    config = SimConfig(ttl=5)
    run = lambda p, pol: replay(val, p, sizes, vocab.cacheable, pol, config)
    none = run(None, PrefetchPolicy(0))
    pop = run(PopularityPredictor().fit(train, len(vocab)), PrefetchPolicy(1))
    markov = run(MarkovPredictor().fit(train, len(vocab)), PrefetchPolicy(1))
    assert none.prefetches == 0
    assert markov.precision > pop.precision
    assert markov.hit_ratio > none.hit_ratio


def test_timestamp_replay_matches_invariants():
    raw = make_synthetic(n_users=150, with_times=True, seed=5)
    vocab = Vocab.fit(raw)
    sessions = vocab.encode(raw)
    sizes = lognormal_sizes(len(vocab), seed=1)
    pred = MarkovPredictor().fit(sessions, len(vocab))
    bucket = TokenBucket(rate=200.0, capacity=float(sizes.max()), start_time=0.0)
    m = replay(sessions, pred, sizes, vocab.cacheable, PrefetchPolicy(2, 0.2), SimConfig(ttl=60), bucket)
    assert m.prefetched_bytes == bucket.granted_bytes == m.useful_bytes + m.wasted_bytes
    assert m.budget_skipped > 0


def test_mean_ci95():
    mean, ci = mean_ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0 and abs(ci - 2.776 * (2.5**0.5) / 5**0.5) < 1e-9


def test_linked_gate_respects_budget_and_balances():
    vocab, train, val = _fitted()
    sizes = lognormal_sizes(len(vocab), seed=0)
    pred = MarkovPredictor().fit(train, len(vocab))
    config = SimConfig(ttl=5)
    ref = reference_bytes(val, pred, sizes, vocab.cacheable, 1, config)
    linked = budget_sweep(val, pred, sizes, vocab.cacheable, BudgetLinkedPolicy(1), config)
    ungated = budget_sweep(val, pred, sizes, vocab.cacheable, PrefetchPolicy(1), config)
    n = sum(len(s.endpoints) for s in val)
    for fraction, m in linked.items():
        assert m.prefetched_bytes == m.useful_bytes + m.wasted_bytes
        rate = fraction * ref / n
        assert m.prefetched_bytes <= rate * n + max(rate * 20, sizes.max()) + 1e-6
    assert linked[0.1].useful_per_mb > ungated[0.1].useful_per_mb


def test_paired_differences_pair_within_run():
    from prefetchcache.experiments.compare import paired_differences

    rows = [
        {"predictor": "lstm", "policy": p, "budget": "50%", "seed": str(s), "useful_per_mb": str(v)}
        for s, (a, b) in enumerate([(100, 110), (200, 212), (50, 55)])
        for p, v in (("top1/ungated", a), ("top1/linked", b))
    ]
    diffs = paired_differences(rows, "top1/linked", "top1/ungated", "useful_per_mb")
    assert diffs == {("lstm", "50%"): [10.0, 12.0, 5.0]}
