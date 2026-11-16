"""Decision-latency benchmark (NFR1: 5 ms or less at p95).

    python -m prefetchcache.experiments.latency artifacts/synthetic/lstm/seed0 artifacts/synthetic/transformer/seed0
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ..predictors.onnx_predictor import OnnxPredictor


def benchmark(predictor, n_endpoints: int, decisions: int = 2000, warmup: int = 200, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    histories = [tuple(rng.integers(0, n_endpoints, size=rng.integers(1, 30)).tolist()) for _ in range(decisions + warmup)]
    for h in histories[:warmup]:
        predictor.predict(h, 10)
    times = []
    for h in histories[warmup:]:
        t0 = time.perf_counter()
        predictor.predict(h, 10)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)), "max_ms": float(max(times))}


def main() -> None:
    for model_dir in sys.argv[1:]:
        p = OnnxPredictor(model_dir)
        runs = [benchmark(p, p.meta["n_endpoints"], seed=s) for s in range(3)]
        p95 = max(r["p95_ms"] for r in runs)
        print(f"{p.name:<12} params={p.meta['parameters']:>8}  p50={np.mean([r['p50_ms'] for r in runs]):.3f} ms  "
              f"worst p95 of 3 runs={p95:.3f} ms  {'PASS' if p95 <= 5 else 'FAIL'} (limit 5 ms)")


if __name__ == "__main__":
    main()
