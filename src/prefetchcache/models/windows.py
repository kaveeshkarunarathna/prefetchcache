"""Turn sessions into fixed-length, left-padded training windows."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.sessions import Session

IGNORE = -100  # target value ignored by the loss


def encode_history(history: Sequence[int], max_len: int, pad_id: int) -> np.ndarray:
    """Last ``max_len`` requests, left-padded. Used for training and serving alike."""
    row = np.full(max_len, pad_id, dtype=np.int64)
    tail = history[-max_len:]
    if tail:
        row[max_len - len(tail) :] = tail
    return row


def make_windows(sessions: Sequence[Session], max_len: int, pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Inputs x[i, t] and next-request targets y[i, t]; padded targets are IGNORE."""
    xs, ys = [], []
    for s in sessions:
        eps = s.endpoints
        for end in range(len(eps) - 1, 0, -max_len):  # windows ending at the last transition, stepping back
            start = max(0, end - max_len)
            x = encode_history(eps[start:end], max_len, pad_id)
            y = np.full(max_len, IGNORE, dtype=np.int64)
            y[max_len - (end - start) :] = eps[start + 1 : end + 1]
            xs.append(x)
            ys.append(y)
    return np.stack(xs), np.stack(ys)
