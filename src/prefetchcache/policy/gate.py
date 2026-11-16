"""Prefetch decision policy: confidence gate (FR2) followed by the byte budget (FR3)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefetchPolicy:
    """``threshold=None`` is ungated top-k; ``top_k=0`` disables prefetching."""

    top_k: int = 1
    threshold: float | None = None

    @property
    def label(self) -> str:
        if self.top_k == 0:
            return "no-prefetch"
        gate = "ungated" if self.threshold is None else f"gate>={self.threshold:g}"
        return f"top{self.top_k}/{gate}"

    def fresh(self) -> "PrefetchPolicy":
        return self  # stateless

    def passes(self, probability: float) -> bool:
        return self.threshold is None or probability >= self.threshold
