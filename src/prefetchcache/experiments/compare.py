"""Paired comparison of two policies across runs.

Each run fixes the model seed and the response sizes, so the difference
between two policies within a run is free of that run-to-run variation.

    python -m prefetchcache.experiments.compare results/coffee/models_validation_all_users_runs.csv \\
        --policy top1/linked --baseline top1/ungated
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from ..sim.metrics import mean_ci95


def paired_differences(rows: list[dict], policy: str, baseline: str, metric: str) -> dict[tuple[str, str], list[float]]:
    """(predictor, budget) -> per-run value of ``policy`` minus ``baseline``."""
    values: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        values[(r["predictor"], r["budget"], r["seed"])][r["policy"]] = float(r[metric])
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (predictor, budget, _seed), by_policy in sorted(values.items(), key=lambda kv: (kv[0][0], kv[0][1], int(kv[0][2]))):
        if policy in by_policy and baseline in by_policy:
            out[(predictor, budget)].append(by_policy[policy] - by_policy[baseline])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_csv")
    ap.add_argument("--policy", default="top1/linked")
    ap.add_argument("--baseline", default="top1/ungated")
    ap.add_argument("--metrics", nargs="*", default=["useful_per_mb", "wasted_mb", "hit_ratio"])
    args = ap.parse_args()
    with open(args.runs_csv) as fh:
        rows = list(csv.DictReader(fh))
    print(f"{args.policy} minus {args.baseline}: mean difference over runs, 95% CI, runs where the policy is higher")
    for metric in args.metrics:
        print(f"\n{metric}")
        diffs = paired_differences(rows, args.policy, args.baseline, metric)
        for (predictor, budget), d in sorted(diffs.items(), key=lambda kv: (kv[0][0], -float(kv[0][1].rstrip("%")))):
            mean, ci = mean_ci95(d)
            verdict = "significant" if abs(mean) > ci and len(d) > 1 else "not significant"
            print(f"  {predictor:<12}{budget:>5}  {mean:>+10.4f} ± {ci:<9.4f} {sum(x > 0 for x in d)}/{len(d)}  {verdict}")


if __name__ == "__main__":
    main()
