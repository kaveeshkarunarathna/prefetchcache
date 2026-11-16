"""Precision-per-byte metrics (deck slide 14)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

MB = 1_000_000


@dataclass
class Metrics:
    requests: int = 0
    cacheable_requests: int = 0
    hits: int = 0
    backend_bytes: int = 0  # bytes fetched from the backend on demand (misses)
    prefetches: int = 0
    prefetched_bytes: int = 0
    useful_prefetches: int = 0  # requested by the same session before expiry
    useful_bytes: int = 0
    wasted_prefetches: int = 0
    wasted_bytes: int = 0  # prefetched but not requested before expiry
    gate_skipped: int = 0
    budget_skipped: int = 0
    already_cached: int = 0
    decision_ms: list[float] = field(default_factory=list, repr=False)

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.cacheable_requests if self.cacheable_requests else 0.0

    @property
    def precision(self) -> float:
        return self.useful_prefetches / self.prefetches if self.prefetches else 0.0

    @property
    def useful_per_mb(self) -> float:
        return self.useful_prefetches / (self.prefetched_bytes / MB) if self.prefetched_bytes else 0.0

    @property
    def byte_precision(self) -> float:
        return self.useful_bytes / self.prefetched_bytes if self.prefetched_bytes else 0.0

    @property
    def decision_p95_ms(self) -> float:
        return float(np.percentile(self.decision_ms, 95)) if self.decision_ms else 0.0

    def summary(self) -> dict[str, float]:
        out = {k: v for k, v in asdict(self).items() if k != "decision_ms"}
        out.update(
            hit_ratio=self.hit_ratio,
            precision=self.precision,
            useful_per_mb=self.useful_per_mb,
            byte_precision=self.byte_precision,
            wasted_mb=self.wasted_bytes / MB,
            prefetched_mb=self.prefetched_bytes / MB,
            decision_p95_ms=self.decision_p95_ms,
        )
        return out


# Two-sided 95% Student t critical values by degrees of freedom.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def mean_ci95(values: list[float]) -> tuple[float, float]:
    """Mean and 95% confidence half-width over seeded runs."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, _T95.get(n - 1, 1.96) * sd / math.sqrt(n)
