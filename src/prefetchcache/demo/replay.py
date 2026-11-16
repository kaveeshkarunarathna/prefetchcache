"""Replay sessions against a running demo app over HTTP.

    python -m prefetchcache.demo.replay --url http://localhost:8000 --sessions 300 --concurrency 20
    python -m prefetchcache.demo.replay --dataset coffee --path data/raw/dataset_session-based_isoweekofday_hour

Without a dataset, sessions come from the synthetic generator. With the
coffee-shop file, validation-split sessions are replayed. Prints response-time
percentiles split by cache hit and miss.
"""
from __future__ import annotations

import argparse
import asyncio
import time

import httpx
import numpy as np

from ..data.synthetic import make_synthetic


def load_sessions(dataset: str, path: str | None, limit: int, seed: int) -> list[tuple[str, list[str]]]:
    if dataset == "coffee":
        from ..data.coffee import load_coffee
        from ..data.split import split_sessions

        raw = split_sessions(load_coffee(path), seed=0, manifest="results/coffee/split_manifest.json").val
    else:
        raw = make_synthetic(n_users=max(50, limit), seed=seed)
    order = np.random.default_rng(seed).permutation(len(raw))[:limit]
    return [(raw[i].session_id, [e.split(" ", 1)[1] for e in raw[i].endpoints]) for i in order]


async def run(url: str, sessions, concurrency: int, think_ms: float, tag: str) -> dict:
    hit_ms: list[float] = []
    miss_ms: list[float] = []
    queue: asyncio.Queue = asyncio.Queue()
    for item in sessions:
        queue.put_nowait(item)

    async def worker(client: httpx.AsyncClient) -> None:
        while not queue.empty():
            session_id, paths = queue.get_nowait()
            for path in paths:
                started = time.perf_counter()
                response = await client.get(path, headers={"x-session-id": f"{tag}-{session_id}"})
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code == 200:
                    (hit_ms if response.headers.get("x-prefetchcache") == "hit" else miss_ms).append(elapsed)
                await asyncio.sleep(think_ms / 1000)

    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        stats = (await client.get("/prefetchcache/stats")).json()
    all_ms = hit_ms + miss_ms
    pct = lambda xs, q: float(np.percentile(xs, q)) if xs else float("nan")
    return {
        "requests": len(all_ms),
        "client_hit_ratio": len(hit_ms) / max(len(all_ms), 1),
        "p50_ms": pct(all_ms, 50),
        "p95_ms": pct(all_ms, 95),
        "hit_p50_ms": pct(hit_ms, 50),
        "miss_p50_ms": pct(miss_ms, 50),
        "server": stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--dataset", choices=["synthetic", "coffee"], default="synthetic")
    ap.add_argument("--path")
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--think-ms", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=str(int(time.time())))
    args = ap.parse_args()
    result = asyncio.run(run(args.url, load_sessions(args.dataset, args.path, args.sessions, args.seed), args.concurrency, args.think_ms, args.tag))
    server = result.pop("server")
    for key, value in result.items():
        print(f"{key:<18}{value:.3f}" if isinstance(value, float) else f"{key:<18}{value}")
    for key in ("hit_ratio", "prefetches", "useful_prefetches", "wasted_prefetches", "precision", "useful_per_mb", "prefetched_mb", "wasted_mb", "decision_p95_ms", "prefetch_errors"):
        value = server[key]
        print(f"server {key:<20}{value:.3f}" if isinstance(value, float) else f"server {key:<20}{value}")


if __name__ == "__main__":
    main()
