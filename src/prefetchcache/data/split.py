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

    If ``manifest`` exists, the split is read from it and never recomputed, so
    the test set stays frozen. If it does not exist, it is written once.
    """
    path = Path(manifest) if manifest else None
    if path and path.exists():
        saved = json.loads(path.read_text())
        assignment = saved["users"]
        if _digest(assignment) != saved["sha256"]:
            raise ValueError(f"split manifest {path} has been modified")
        known = {u for part in assignment.values() for u in part}
        missing = {s.user_id for s in sessions} - known
        if missing:
            raise ValueError(f"{len(missing)} users are not in the frozen split manifest")
    else:
        assignment = split_users([s.user_id for s in sessions], seed)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"seed": seed, "sha256": _digest(assignment), "users": assignment}))
    part_of = {u: part for part, users in assignment.items() for u in users}
    out: dict[str, list[RawSession]] = {"train": [], "val": [], "test": []}
    for s in sessions:
        out[part_of[s.user_id]].append(s)
    return Split(out["train"], out["val"], out["test"])
