"""Predictor interface: history of endpoint ids in, ranked next-endpoint probabilities out."""
from __future__ import annotations

from typing import Protocol, Sequence

from ..data.sessions import Session

Ranked = list[tuple[int, float]]  # (endpoint id, probability), highest first


class Predictor(Protocol):
    name: str

    def fit(self, sessions: Sequence[Session], n_endpoints: int) -> "Predictor": ...

    def predict(self, history: Sequence[int], top_n: int = 5) -> Ranked: ...
