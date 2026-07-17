from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    input_dir: Path | None = None
    output_dir: Path | None = None
    worker_count_resolved: int | None = None
    device_resolved: str | None = None
    temporary_output_path: Path | None = None
    diagnostic_capture: bool = False
    max_files: int | None = None
    cancel_token: object | None = None
    progress_callback: Callable[[object], None] | None = None


@dataclass(frozen=True, slots=True)
class DeveloperOverrides:
    enabled: bool = False
    values: Mapping[str, object] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.values)
