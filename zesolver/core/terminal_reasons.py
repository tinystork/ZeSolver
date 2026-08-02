from __future__ import annotations

from enum import Enum


class TerminalReasonCode(str, Enum):
    NEAR_UNRESOLVED_BLIND_UNAVAILABLE = "NEAR_UNRESOLVED_BLIND_UNAVAILABLE"
    ALL_ENABLED_SOLVERS_EXHAUSTED = "ALL_ENABLED_SOLVERS_EXHAUSTED"
    INPUT_UNREADABLE = "INPUT_UNREADABLE"
    INPUT_MISSING = "INPUT_MISSING"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    WRITE_ERROR = "WRITE_ERROR"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    SKIPPED_EXISTING_WCS = "SKIPPED_EXISTING_WCS"
    CANCELLED = "CANCELLED"


ELIGIBLE_UNRESOLVED_REASON_CODES = frozenset(
    {
        TerminalReasonCode.NEAR_UNRESOLVED_BLIND_UNAVAILABLE.value,
        TerminalReasonCode.ALL_ENABLED_SOLVERS_EXHAUSTED.value,
    }
)


def normalize_terminal_reason_code(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().upper()
    if not raw:
        return None
    try:
        return TerminalReasonCode(raw).value
    except ValueError:
        return raw


def is_unresolved_move_eligible(reason_code: object) -> bool:
    return normalize_terminal_reason_code(reason_code) in ELIGIBLE_UNRESOLVED_REASON_CODES

