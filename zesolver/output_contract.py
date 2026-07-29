from __future__ import annotations

from pathlib import Path


UNRESOLVED_DIRECTORY_NAME = "unresolved_by_zesolver"


def is_inside_unresolved_directory(path: Path) -> bool:
    return any(part == UNRESOLVED_DIRECTORY_NAME for part in Path(path).parts)

