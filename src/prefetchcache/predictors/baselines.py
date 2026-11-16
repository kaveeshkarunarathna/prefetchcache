"""Non-neural baselines: popularity and first-order Markov."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.sessions import Session
from .base import Ranked


def _ranked(probs: np.ndarray, max_n: int) -> Ranked:
    order = np.argsort(-probs, kind="stable")[:max_n]
    return [(int(i), float(probs[i])) for i in order if probs[i] > 0]


class PopularityPredictor:
    """Always predicts the most frequently requested endpoints."""

    name = "popularity"
    MAX_N = 20

    def fit(self, sessions: Sequence[Session], n_endpoints: int) -> "PopularityPredictor":
        counts = np.zeros(n_endpoints)
        for s in sessions:
            np.add.at(counts, list(s.endpoints), 1)
        self._ranked = _ranked(counts / max(counts.sum(), 1), self.MAX_N)
        return self

    def predict(self, history: Sequence[int], top_n: int = 5) -> Ranked:
        return self._ranked[:top_n]


class MarkovPredictor:
    """First-order Markov chain; backs off to popularity for unseen states."""

    name = "markov1"
    MAX_N = 20

    def fit(self, sessions: Sequence[Session], n_endpoints: int) -> "MarkovPredictor":
        counts: dict[int, dict[int, int]] = {}
        for s in sessions:
            eps = s.endpoints
            for a, b in zip(eps, eps[1:]):
                row = counts.setdefault(a, {})
                row[b] = row.get(b, 0) + 1
        self._table: dict[int, Ranked] = {}
        for a, row in counts.items():
            total = sum(row.values())
            top = sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))[: self.MAX_N]
            self._table[a] = [(b, c / total) for b, c in top]
        self._fallback = PopularityPredictor().fit(sessions, n_endpoints)
        return self

    def predict(self, history: Sequence[int], top_n: int = 5) -> Ranked:
        if not history:
            return self._fallback.predict(history, top_n)
        ranked = self._table.get(history[-1])
        if ranked is None:
            return self._fallback.predict(history, top_n)
        return ranked[:top_n]
