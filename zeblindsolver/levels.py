from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class QuadLevelSpec:
    name: str
    min_area: float
    max_area: float
    min_diameter: float | None
    max_diameter: float | None
    bucket_cap: int

    def to_manifest(self) -> dict[str, float | int | None]:
        return {
            "name": self.name,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "min_diameter": self.min_diameter,
            "max_diameter": self.max_diameter,
            "bucket_cap": self.bucket_cap,
        }


LEVEL_SPECS: Sequence[QuadLevelSpec] = (
    QuadLevelSpec(name="L", min_area=0.5, max_area=16.0, min_diameter=0.2, max_diameter=6.0, bucket_cap=8192),
    QuadLevelSpec(name="M", min_area=0.15, max_area=4.0, min_diameter=0.1, max_diameter=3.5, bucket_cap=4096),
    QuadLevelSpec(name="S", min_area=0.01, max_area=1.5, min_diameter=0.05, max_diameter=2.0, bucket_cap=2048),
)

LEVEL_MAP: dict[str, QuadLevelSpec] = {level.name: level for level in LEVEL_SPECS}
