"""Dataset loading shared by training and sweep scripts."""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

from ..data.coffee import load_coffee
from ..data.retailrocket import load_retailrocket
from ..data.sessions import Session, Vocab
from ..data.split import split_sessions
from ..data.synthetic import make_synthetic


@dataclass
class Dataset:
    name: str
    vocab: Vocab
    train: list[Session]
    val: list[Session]
    n_test_sessions: int  # the test split itself is only loaded by the final experiments


def keep_user(user_id: str, fraction: float) -> bool:
    """Deterministic user subsample for development runs."""
    return fraction >= 1.0 or zlib.crc32(user_id.encode()) % 10_000 < fraction * 10_000


def load_dataset(name: str, path: str | None, min_count: int, results_dir: str | Path, user_sample: float = 1.0, val_sample: float = 1.0) -> Dataset:
    """Load, split by user against the frozen manifest, then optionally subsample users.

    ``user_sample`` thins training and validation users alike; ``val_sample``
    thins validation users only, so models trained on all users can be swept
    on part of the validation split with the same vocabulary.
    """
    if name == "synthetic":
        raw = make_synthetic(seed=0)
    elif name == "coffee":
        raw = load_coffee(path)
    elif name == "retailrocket":
        raw = load_retailrocket(path)
    else:
        raise ValueError(name)
    manifest = Path(results_dir) / name / "split_manifest.json"
    split = split_sessions(raw, seed=0, manifest=manifest)
    train = [s for s in split.train if keep_user(s.user_id, user_sample)]
    val = [s for s in split.val if keep_user(s.user_id, min(user_sample, val_sample))]
    vocab = Vocab.fit(train, min_count=min_count)
    return Dataset(name, vocab, vocab.encode(train), vocab.encode(val), len(split.test))
