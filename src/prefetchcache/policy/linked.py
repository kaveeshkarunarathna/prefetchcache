"""Budget-linked gate: the confidence a prefetch needs depends on its size
and on how much of the byte budget is left.

Spending a byte budget on prefetches that will be used is a knapsack problem:
take the candidates with the highest probability of use per byte. The policy
keeps a price in "probability per byte" and admits a candidate only if
``probability / size >= price``. The price rises while the token bucket is
below its target fill and falls while it is above, so the gate tightens
exactly when the budget is scarce and opens when bytes would go unused.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass
class BudgetLinkedPolicy:
    top_k: int = 1
    min_confidence: float = 0.0  # absolute floor, regardless of budget
    reference_bytes: float = 8_000.0  # typical response size; sets the starting price
    target_fill: float = 0.5
    step: float = 0.05
    price: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.price <= 0:
            self.price = 0.5 / self.reference_bytes

    @property
    def label(self) -> str:
        floor = f",p>={self.min_confidence:g}" if self.min_confidence else ""
        return f"top{self.top_k}/linked{floor}"

    def fresh(self) -> "BudgetLinkedPolicy":
        """A copy with the price reset, for a new replay."""
        return replace(self, price=0.0)

    def observe(self, fill: float) -> None:
        """Update the price from the bucket fill level (0 empty, 1 full)."""
        self.price *= math.exp(self.step * (self.target_fill - fill))

    def admit(self, probability: float, size: int) -> bool:
        return probability >= self.min_confidence and probability / size >= self.price
