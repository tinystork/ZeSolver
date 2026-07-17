from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class PipelineTelemetry:
    request_id: str | None
    pipeline_profile: str
    near_profile: str
    blind_profile: str
    started_at: float = field(default_factory=time.perf_counter)
    catalog_source: str | None = None
    catalog_status: str | None = None
    catalog_coverage_fraction: float | None = None
    near_attempted: bool = False
    near_result: str | None = None
    blind_attempted: bool = False
    blind_result: str | None = None
    final_status: str | None = None
    wcs_written: bool = False
    warnings: list[str] = field(default_factory=list)

    def finish(self, *, final_status: str, wcs_written: bool) -> Mapping[str, Any]:
        self.final_status = final_status
        self.wcs_written = bool(wcs_written)
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "pipeline_profile": self.pipeline_profile,
            "near_profile": self.near_profile,
            "blind_profile": self.blind_profile,
            "catalog_source": self.catalog_source,
            "catalog_status": self.catalog_status,
            "catalog_coverage_fraction": self.catalog_coverage_fraction,
            "near_attempted": self.near_attempted,
            "near_result": self.near_result,
            "blind_attempted": self.blind_attempted,
            "blind_result": self.blind_result,
            "final_status": self.final_status,
            "wcs_written": self.wcs_written,
            "duration_s": round(time.perf_counter() - self.started_at, 6),
            "warnings": tuple(dict.fromkeys(self.warnings)),
        }
