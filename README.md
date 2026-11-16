# PrefetchCache

Confidence-gated, byte-budgeted server-side prefetching middleware for web
application caches, with a precision-per-byte evaluation method.

Final year project, 6SENG010C. Kaveesh (IIT 20223173 | UoW W1986691).
Supervisor: Ms Sapna Kumarapathirage.

## Status

| Milestone | Work package | Contents | State |
|---|---|---|---|
| M1 | WP2 | Data loaders, user-level split, baselines, replay simulator, precision-per-byte metrics, budget sweep | Done (synthetic data only) |
| M2 | WP2 | Real coffee-shop and Retailrocket data: loader checks, EDA, frozen test split | Waiting on dataset files |
| M3 | WP3 | LSTM and small Transformer in PyTorch, temperature scaling, ONNX export, latency benchmark (NFR1) | Next |
| M4 | WP4 | Gate linked to the token bucket, FastAPI middleware, Redis key scheme, prefetch log, admin API (FR2 to FR5, NFR2, NFR3) | |
| M5 | WP5 | Demo app, measured response sizes, React dashboard, Docker Compose, Locust load test (O4) | |
| M6 | WP6 | RQ1 to RQ3 experiments on the frozen test split, five seeds, 95% confidence intervals | |

## Layout

```
src/prefetchcache/
  data/          sessions.py   RawSession, Session, Vocab (fitted on training data only)
                 coffee.py     SPMF loader for the coffee-shop clickstreams
                 retailrocket.py  events.csv loader, 30-minute session gap
                 synthetic.py  Markov-chain generator for tests
                 split.py      70/15/15 split by user, frozen manifest with checksum
                 sizes.py      response sizes per endpoint
  predictors/    base.py       predictor interface (history in, ranked probabilities out)
                 baselines.py  popularity, first-order Markov
  policy/        gate.py       confidence threshold and top-k
                 budget.py     token-bucket byte budget
  sim/           cache.py      per-session keys, TTL, LRU byte capacity
                 replay.py     request flow of deck slide 8
                 metrics.py    useful prefetches per MB, wasted bytes, precision, hit ratio
                 sweep.py      replay at five byte budgets
  experiments/   run_baselines.py
tests/           pytest suite; runs in GitHub Actions on every push
```

Planned: `models/` (LSTM, Transformer, calibration, ONNX export), `middleware/`
(FastAPI + Redis), `demo/` (shop API and dashboard), `loadtest/` (Locust).

## Run

```
pip install -e ".[dev]"
pytest -q
python -m prefetchcache.experiments.run_baselines --dataset synthetic
python -m prefetchcache.experiments.run_baselines --dataset retailrocket --path data/raw/events.csv
```

Datasets are not stored in the repository. Place them under `data/raw/`.

## Metric definitions

- **Useful prefetch**: a prefetched response requested by the same session before its TTL expires. Counted once.
- **Wasted bytes**: bytes prefetched and not requested before expiry, session end or eviction.
- **Useful prefetches per MB**: useful prefetches divided by MB prefetched.
- **Precision**: useful prefetches divided by prefetches issued.
- **Hit ratio**: cacheable (GET) requests served from the cache, divided by cacheable requests.

Prefetched bytes always equal useful bytes plus wasted bytes; the test suite checks this.

## Decisions to confirm with the supervisor

These are not fixed in the proposal deck. The code uses the proposed value for each until confirmed.

| Item | Proposed | Where |
|---|---|---|
| Five budget levels | 100, 50, 25, 10 and 5% of the bytes spent by ungated top-k with the same model | `sim/sweep.py` |
| Token-bucket burst | 20 time units of refill, and at least the largest response | `sim/sweep.py` |
| Clock for the coffee-shop data (no timestamps) | One tick per request; TTL and refill are counted in requests | `sim/replay.py` |
| Cache expiry | 5 requests (coffee shop); 300 s (Retailrocket) | `run_baselines.py --ttl` |
| Top-k | k = 1, with k = 3 as a sensitivity run | `run_baselines.py --top-k` |
| Retailrocket endpoints | view = `GET /items/{id}`, add to cart = `POST /cart`, transaction = `POST /orders`; items seen fewer than 5 times in training map to an unknown id and are never prefetched | `data/retailrocket.py` |
| Response sizes until the demo app exists | Seeded log-normal, median 8 kB | `data/sizes.py` |
| Success criteria | The gated policy gives more useful prefetches per MB than ungated top-k at every budget, with at most 10% relative loss in hit ratio; decision time at or below 5 ms p95 | to be fixed before testing |

## Known limitations of M1

- No run has used real data yet. The coffee-shop loader parses standard SPMF but has not been checked against the real file; where user ids and day/hour sit must be confirmed.
- Results under `results/baselines/synthetic/` come from generated sessions and placeholder sizes. They show the pipeline works; they are not findings.
- The gate and the token bucket are applied one after the other. With a tight bucket only small responses fit, which raises useful prefetches per MB for the ungated policy too. Linking the gate to the budget (WP4) has to account for this, for example by ranking candidates by confidence per byte.
