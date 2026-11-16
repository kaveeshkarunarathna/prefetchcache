"""User-level train/validation/test split, with a frozen test manifest."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .sessions import RawSession


@dataclass(frozen=True)
class Split:
    train: list[RawSession]
    val: list[RawSession]
    test: list[RawSession]


def split_users(
    user_ids: Sequence[str], seed: int = 0, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> dict[str, list[str]]:
    """Assign every user to exactly one of train/val/test."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    users = sorted(set(user_ids))
    order = np.random.default_rng(seed).permutation(len(users))
    n_train = int(round(fractions[0] * len(users)))
    n_val = int(round(fractions[1] * len(users)))
    shuffled = [users[i] for i in order]
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def _digest(assignment: dict[str, list[str]]) -> str:
    payload = json.dumps(assignment, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def split_sessions(
    sessions: Sequence[RawSession], seed: int = 0, manifest: str | Path | None = None
) -> Split:
    """Split sessions by user.

    The split is a deterministic function of the user ids and the seed. The
    manifest records the seed, the size of each part and a checksum of the
    assignment. If it exists, its seed is used and the recomputed split must
    match the checksum, so the test set cannot change unnoticed.
    """
    path = Path(manifest) if manifest else None
    saved = json.loads(path.read_text()) if path and path.exists() else None
    if saved:
        seed = saved["seed"]
    assignment = split_users([s.user_id for s in sessions], seed)
    if saved:
        if _digest(assignment) != saved["sha256"]:
            raise ValueError(f"the split no longer matches the frozen manifest {path}")
    elif path:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"seed": seed, "users": {k: len(v) for k, v in assignment.items()}, "sha256": _digest(assignment)}
        path.write_text(json.dumps(record, indent=1) + "\n")
    part_of = {u: part for part, users in assignment.items() for u in users}
    out: dict[str, list[RawSession]] = {"train": [], "val": [], "test": []}
    for s in sessions:
        out[part_of[s.user_id]].append(s)
    return Split(out["train"], out["val"], out["test"])
