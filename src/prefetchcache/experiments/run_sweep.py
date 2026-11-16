"""Budget sweep on the validation split: baselines, and trained models if exported.

    python -m prefetchcache.experiments.run_sweep --dataset synthetic
    python -m prefetchcache.experiments.run_sweep --dataset synthetic --models-dir artifacts
    python -m prefetchcache.experiments.run_sweep --dataset retailrocket --path data/raw/events.csv

The frozen test split is not touched here. Seed s uses the model trained with
seed s and, until sizes measured from the demo app are available, the
response-size draw s.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..data.sizes import lognormal_sizes
from ..policy.gate import PrefetchPolicy
from ..policy.linked import BudgetLinkedPolicy
from ..predictors.baselines import MarkovPredictor, PopularityPredictor
from ..sim.metrics import mean_ci95
from ..sim.replay import SimConfig, replay, uses_timestamps
from .common import load_dataset
from ..sim.sweep import DEFAULT_BUDGET_FRACTIONS, budget_sweep

class MemoPredictor:
    """Caches predictions by history, so each policy and budget reuses them."""

    def __init__(self, inner):
        self.inner, self.name, self._memo = inner, inner.name, {}

    def predict(self, history, top_n=5):
        key = (tuple(history[-self.window :]), top_n)
        hit = self._memo.get(key)
        if hit is None:
            hit = self._memo[key] = self.inner.predict(history, top_n)
        return hit

    @property
    def window(self) -> int:
        return getattr(self.inner, "max_len", 1)


REPORTED = ("useful_per_mb", "precision", "wasted_mb", "prefetched_mb", "hit_ratio")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "coffee", "retailrocket"], default="synthetic")
    ap.add_argument("--path")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--thresholds", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    ap.add_argument("--ttl", type=float, help="seconds with timestamps, requests otherwise")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--user-sample", type=float, default=1.0, help="fraction of users to keep (development runs)")
    ap.add_argument("--models-dir", help="directory holding <dataset>/<model>/seed<N> ONNX exports")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.dataset
    data = load_dataset(args.dataset, args.path, args.min_count, args.out, args.user_sample)
    vocab, train, val = data.vocab, data.train, data.val
    ttl = args.ttl if args.ttl is not None else (300.0 if uses_timestamps(val) else 5.0)
    config = SimConfig(ttl=ttl)
    print(f"{args.dataset}: {len(vocab)} endpoints, train/val/test = "
          f"{len(train)}/{len(val)}/{data.n_test_sessions} sessions, ttl={ttl:g}")

    baselines = [PopularityPredictor().fit(train, len(vocab)), MarkovPredictor().fit(train, len(vocab))]
    policies = [PrefetchPolicy(args.top_k, None)] + [PrefetchPolicy(args.top_k, t) for t in args.thresholds]
    policies.append(BudgetLinkedPolicy(args.top_k))

    # runs[(predictor, policy, budget)][metric] -> one value per seed
    runs: dict[tuple[str, str, str], dict[str, list[float]]] = {}

    def record(key, metrics) -> None:
        summary = metrics.summary()
        slot = runs.setdefault(key, {k: [] for k in REPORTED})
        for k in REPORTED:
            slot[k].append(summary[k])

    for seed in range(args.seeds):
        sizes = lognormal_sizes(len(vocab), seed=seed)
        record(("none", "no-prefetch", "-"), replay(val, None, sizes, vocab.cacheable, PrefetchPolicy(0), config))
        predictors = list(baselines)
        if args.models_dir:
            from ..predictors.onnx_predictor import OnnxPredictor

            for kind in ("lstm", "transformer"):
                model_dir = Path(args.models_dir) / args.dataset / kind / f"seed{seed}"
                if model_dir.exists():
                    predictors.append(MemoPredictor(OnnxPredictor(model_dir)))
        for pred in predictors:
            for policy in policies:
                sweep = budget_sweep(val, pred, sizes, vocab.cacheable, policy, config, DEFAULT_BUDGET_FRACTIONS)
                for fraction, metrics in sweep.items():
                    record((pred.name, policy.label, f"{fraction:.0%}"), metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_name = "models_validation.csv" if args.models_dir else "baselines_validation.csv"
    with open(out_dir / csv_name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["predictor", "policy", "budget", *[f"{k}_{s}" for k in REPORTED for s in ("mean", "ci95")]])
        for key, vals in runs.items():
            row = list(key)
            for k in REPORTED:
                mean, ci = mean_ci95(vals[k])
                row += [f"{mean:.4f}", f"{ci:.4f}"]
            w.writerow(row)
    print(f"wrote {out_dir / csv_name}")

    print(f"\n{'predictor':<13}{'policy':<16}{'budget':>7}{'useful/MB':>11}{'precision':>11}{'wasted MB':>11}{'hit ratio':>11}")
    for (pred, pol, bud), vals in runs.items():
        print(f"{pred:<13}{pol:<16}{bud:>7}{mean_ci95(vals['useful_per_mb'])[0]:>11.1f}{mean_ci95(vals['precision'])[0]:>11.3f}"
              f"{mean_ci95(vals['wasted_mb'])[0]:>11.2f}{mean_ci95(vals['hit_ratio'])[0]:>11.3f}")


if __name__ == "__main__":
    main()
