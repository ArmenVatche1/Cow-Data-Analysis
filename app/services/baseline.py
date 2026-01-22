from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable

@dataclass(frozen=True)
class BaselineRow:
    breed: str
    dim_min: int
    dim_max: int
    expected_liters: float
    std_dev: Optional[float] = None

class BaselineTable:
    def __init__(self, rows: Iterable[BaselineRow]):
        self.rows = list(rows)

    def lookup(self, breed: str, dim: int) -> float:
        candidates = [r for r in self.rows if r.breed == breed and r.dim_min <= dim <= r.dim_max]
        if candidates:
            return candidates[0].expected_liters

        candidates = [r for r in self.rows if r.breed == "ALL" and r.dim_min <= dim <= r.dim_max]
        if candidates:
            return candidates[0].expected_liters

        return 0.0
