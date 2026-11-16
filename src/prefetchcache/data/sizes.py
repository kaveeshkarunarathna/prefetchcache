"""Response sizes in bytes per endpoint.

Neither dataset records response sizes. The final sizes are measured from the
demo app; until then a seeded log-normal stands in, and the same class is
used for the planned sensitivity analysis over size distributions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .sessions import Vocab


def lognormal_sizes(n_endpoints: int, seed: int = 0, median_bytes: float = 8_000.0, sigma: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sizes = rng.lognormal(np.log(median_bytes), sigma, size=n_endpoints)
    return np.maximum(sizes, 200).astype(np.int64)


def measured_sizes(vocab: Vocab, path: str | Path, default_bytes: int) -> np.ndarray:
    """Sizes measured from the demo app, as a JSON map of endpoint -> bytes."""
    table = json.loads(Path(path).read_text())
    return np.array([int(table.get(name, default_bytes)) for name in vocab.names], dtype=np.int64)
