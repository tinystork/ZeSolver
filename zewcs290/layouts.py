"""
Utilities for working with the HNSKY/ASTAP tiling schemes.

The numerical data is sourced from:
  * https://www.hnsky.org/astap.htm  (1476 layout)
  * http://www.hnsky.org/help/uk/hnsky.htm  (290 layout)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
import json
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RingDef:
    """Declination ring descriptor coming from ``layouts.json``."""

    ring_label: str
    dec_min_deg: float
    dec_max_deg: float
    ra_cells: int
    dec_step_deg: Optional[float]

    @property
    def span_deg(self) -> float:
        return self.dec_max_deg - self.dec_min_deg


@dataclass(frozen=True)
class LayoutDefinition:
    """In-memory representation of an ASTAP/HNSKY tiling."""

    name: str
    rings: Tuple[RingDef, ...]
    _by_index: Dict[int, RingDef] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.rings:
            raise ValueError(f"layout {self.name} has no rings defined")
        mapping: Dict[int, RingDef] = {}
        for idx, ring in enumerate(self.rings):
            try:
                start = int(ring.ring_label.split("-", 1)[0])
            except (ValueError, IndexError):
                start = idx
            mapping[start] = ring
        object.__setattr__(self, "_by_index", mapping)

    def ring_for_index(self, ring_index: int) -> RingDef:
        try:
            return self._by_index[ring_index]
        except KeyError as exc:
            raise KeyError(f"ring_index {ring_index} out of range for layout {self.name}") from exc

    def iter_rings(self) -> Iterable[RingDef]:
        return iter(self.rings)


def _load_layouts() -> Dict[str, LayoutDefinition]:
    """Read the bundled JSON metadata."""

    with resources.open_text("zewcs290.data", "layouts.json", encoding="utf-8") as handle:
        payload = json.load(handle)

    layouts: Dict[str, LayoutDefinition] = {}
    for name, rows in payload.items():
        rings: List[RingDef] = []
        for row in rows:
            rings.append(
                RingDef(
                    ring_label=row["ring"],
                    dec_min_deg=float(row["dec_min_deg"]),
                    dec_max_deg=float(row["dec_max_deg"]),
                    ra_cells=int(row["ra_cells"]),
                    dec_step_deg=None if row.get("dec_step_deg") is None else float(row["dec_step_deg"]),
                )
            )
        layouts[name] = LayoutDefinition(name=name, rings=tuple(rings))
    return layouts


LAYOUTS: Dict[str, LayoutDefinition] = _load_layouts()


def get_layout(name: str) -> LayoutDefinition:
    try:
        return LAYOUTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown layout {name!r}") from exc


def list_layout_names() -> Tuple[str, ...]:
    return tuple(sorted(LAYOUTS))
