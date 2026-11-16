"""Serve an exported LSTM or Transformer through ONNX Runtime (NFR1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort

from .base import Ranked


class OnnxPredictor:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "model.json").read_text())
        self.name = meta["model"]
        self.meta = meta
        self.max_len: int = meta["max_len"]
        self.pad_id: int = meta["pad_id"]
        self.temperature: float = meta["temperature"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self._buf = np.empty((1, self.max_len), dtype=np.int64)

    def fit(self, sessions, n_endpoints):  # trained offline
        return self

    def predict(self, history: Sequence[int], top_n: int = 5) -> Ranked:
        buf = self._buf
        buf.fill(self.pad_id)
        tail = history[-self.max_len :]
        if tail:
            buf[0, self.max_len - len(tail) :] = tail
        logits = self._session.run(None, {"history": buf})[0][0] / self.temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        n = min(top_n, len(probs))
        top = np.argpartition(-probs, n - 1)[:n]
        top = top[np.argsort(-probs[top])]
        return [(int(i), float(probs[i])) for i in top]
