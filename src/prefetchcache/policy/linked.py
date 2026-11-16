"""Budget-linked gate: the confidence a prefetch needs depends on its size
and on how much of the byte budget is left.

Spending a byte budget on prefetches that will be used is a knapsack problem:
take the candidates with the highest probability of use per byte. The policy
keeps a price in "probability per byte" and admits a candidate only if
``probability / size >= price``. The price rises while the token bucket is
below its target fill and falls while it is above, so the gate tightens
exactly when the budget is scarce and opens when bytes would go unused.

The price moves in proportion to elapsed clock time, not to the number of
decisions, so many concurrent requests in one instant do not overreact.
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
    step: float = 0.05  # relative price change per clock unit when the bucket is fully off target
    price: float = field(default=0.0)
    _last: float | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.price <= 0:
            self.price = 0.5 / self.reference_bytes

    @property
    def label(self) -> str:
        floor = f",p>={self.min_confidence:g}" if self.min_confidence else ""
        return f"top{self.top_k}/linked{floor}"

    def fresh(self) -> "BudgetLinkedPolicy":
        """A copy with the price reset, for a new replay."""
        return replace(self, price=0.0, _last=None)

    def observe(self, fill: float, now: float | None = None) -> None:
        """Update the price from the bucket fill level (0 empty, 1 full).

        ``now`` is the caller's clock (requests in the coffee-shop replay,
        seconds elsewhere). Without it, each call counts as one clock unit.
        """
        elapsed = 1.0 if now is None or self._last is None else max(0.0, now - self._last)
        if now is not None:
            self._last = now
        self.price *= math.exp(self.step * min(elapsed, 100.0) * (self.target_fill - fill))
        self.price = min(max(self.price, 1e-12), 1.0)  # probability per byte cannot usefully exceed 1

    def admit(self, probability: float, size: int) -> bool:
        return probability >= self.min_confidence and probability / size >= self.price
