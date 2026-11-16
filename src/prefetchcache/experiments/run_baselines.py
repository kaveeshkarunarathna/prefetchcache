"""Milestone 1 run: baselines across the budget sweep, on the validation split.

    python -m prefetchcache.experiments.run_baselines --dataset synthetic
    python -m prefetchcache.experiments.run_baselines --dataset retailrocket --path data/raw/events.csv
    python -m prefetchcache.experiments.run_baselines --dataset coffee --path data/raw/<file>

The frozen test split is not touched here. Seeds vary the response-size draw
until sizes measured from the demo app are available.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..data.coffee import load_coffee
from ..data.retailrocket import load_retailrocket
from ..data.sessions import Vocab
from ..data.sizes import lognormal_sizes
from ..data.split import split_sessions
from ..data.synthetic import make_synthetic
from ..policy.gate import PrefetchPolicy
from ..predictors.baselines import MarkovPredictor, PopularityPredictor
from ..sim.metrics import mean_ci95
from ..sim.replay import SimConfig, replay, uses_timestamps
from ..sim.sweep import DEFAULT_BUDGET_FRACTIONS, budget_sweep

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
    ap.add_argument("--out", default="results/baselines")
    args = ap.parse_args()

    if args.dataset == "synthetic":
        raw = make_synthetic(seed=0)
    elif args.dataset == "coffee":
        raw = load_coffee(args.path)
    else:
        raw = load_retailrocket(args.path)

    out_dir = Path(args.out) / args.dataset
    split = split_sessions(raw, seed=0, manifest=out_dir / "split_manifest.json")
    vocab = Vocab.fit(split.train, min_count=args.min_count)
    train, val = vocab.encode(split.train), vocab.encode(split.val)
    ttl = args.ttl if args.ttl is not None else (300.0 if uses_timestamps(val) else 5.0)
    config = SimConfig(ttl=ttl)
    print(f"{args.dataset}: {len(raw)} sessions, {len(vocab)} endpoints, train/val/test = "
          f"{len(split.train)}/{len(split.val)}/{len(split.test)} sessions, ttl={ttl:g}")

    predictors = [PopularityPredictor().fit(train, len(vocab)), MarkovPredictor().fit(train, len(vocab))]
    policies = [PrefetchPolicy(args.top_k, None)] + [PrefetchPolicy(args.top_k, t) for t in args.thresholds]

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
        for pred in predictors:
            for policy in policies:
                sweep = budget_sweep(val, pred, sizes, vocab.cacheable, policy, config, DEFAULT_BUDGET_FRACTIONS)
                for fraction, metrics in sweep.items():
                    record((pred.name, policy.label, f"{fraction:.0%}"), metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baselines_validation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["predictor", "policy", "budget", *[f"{k}_{s}" for k in REPORTED for s in ("mean", "ci95")]])
        for key, vals in runs.items():
            row = list(key)
            for k in REPORTED:
                mean, ci = mean_ci95(vals[k])
                row += [f"{mean:.4f}", f"{ci:.4f}"]
            w.writerow(row)
    print(f"wrote {out_dir / 'baselines_validation.csv'}")

    print(f"\n{'predictor':<11}{'policy':<16}{'budget':>7}{'useful/MB':>11}{'precision':>11}{'wasted MB':>11}{'hit ratio':>11}")
    for (pred, pol, bud), vals in runs.items():
        print(f"{pred:<11}{pol:<16}{bud:>7}{mean_ci95(vals['useful_per_mb'])[0]:>11.1f}{mean_ci95(vals['precision'])[0]:>11.3f}"
              f"{mean_ci95(vals['wasted_mb'])[0]:>11.2f}{mean_ci95(vals['hit_ratio'])[0]:>11.3f}")


if __name__ == "__main__":
    main()
