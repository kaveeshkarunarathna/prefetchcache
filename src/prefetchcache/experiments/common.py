"""Dataset loading shared by training and sweep scripts."""
from __future__ import annotations

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


def load_dataset(name: str, path: str | None, min_count: int, results_dir: str | Path) -> Dataset:
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
    vocab = Vocab.fit(split.train, min_count=min_count)
    return Dataset(name, vocab, vocab.encode(split.train), vocab.encode(split.val), len(split.test))
