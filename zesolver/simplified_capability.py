from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .catalog_resources import SolverCatalogResources
from .settings import ProductSettings


class SimplifiedSolveCapability(str, Enum):
    FULL_LOCAL = "FULL_LOCAL"
    NEAR_ONLY = "NEAR_ONLY"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class SimplifiedCapabilityDecision:
    capability: SimplifiedSolveCapability
    near_available: bool
    blind4d_available: bool
    catalog_source_used: str
    blind4d_disabled_reason: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def effective_blind_enabled(self) -> bool:
        return self.capability is SimplifiedSolveCapability.FULL_LOCAL

    @property
    def preflight_error(self) -> str | None:
        if self.capability is SimplifiedSolveCapability.UNAVAILABLE:
            return "NO_LOCAL_CATALOG_SOURCE_AVAILABLE"
        return None

    def telemetry(self) -> dict[str, object]:
        return {
            "simplified_capability": self.capability.value,
            "near_available": self.near_available,
            "blind4d_available": self.blind4d_available,
            "blind4d_disabled_reason": self.blind4d_disabled_reason,
            "catalog_source_used": self.catalog_source_used,
            "warnings": list(self.warnings),
        }


def is_simplified_interface(product_settings: ProductSettings) -> bool:
    mode = str(getattr(product_settings, "interface_mode", "expert") or "expert").strip().lower()
    return mode in {"easy", "wizard", "simple", "simplified"}


def evaluate_simplified_capability(resources: SolverCatalogResources) -> SimplifiedCapabilityDecision:
    near = bool(resources.near_available)
    blind = bool(resources.blind4d_available)
    warnings = list(resources.warnings or ())
    if resources.source == "library":
        if near and blind:
            return SimplifiedCapabilityDecision(
                capability=SimplifiedSolveCapability.FULL_LOCAL,
                near_available=True,
                blind4d_available=True,
                catalog_source_used="library",
                warnings=tuple(warnings),
            )
        if near:
            return SimplifiedCapabilityDecision(
                capability=SimplifiedSolveCapability.NEAR_ONLY,
                near_available=True,
                blind4d_available=False,
                catalog_source_used="library",
                blind4d_disabled_reason="library_blind4d_unavailable",
                warnings=tuple((*warnings, "catalog_library_partial_near_only")),
            )
        return SimplifiedCapabilityDecision(
            capability=SimplifiedSolveCapability.UNAVAILABLE,
            near_available=False,
            blind4d_available=blind,
            catalog_source_used="library",
            blind4d_disabled_reason=("library_near_unavailable" if blind else "library_unusable"),
            warnings=tuple((*warnings, "catalog_library_missing_near")),
        )
    if near:
        return SimplifiedCapabilityDecision(
            capability=SimplifiedSolveCapability.NEAR_ONLY,
            near_available=True,
            blind4d_available=False,
            catalog_source_used=resources.source or "legacy",
            blind4d_disabled_reason="no_complete_catalog_library",
            warnings=tuple(warnings),
        )
    return SimplifiedCapabilityDecision(
        capability=SimplifiedSolveCapability.UNAVAILABLE,
        near_available=False,
        blind4d_available=False,
        catalog_source_used=resources.source or "none",
        blind4d_disabled_reason="no_complete_catalog_library",
        warnings=tuple(warnings),
    )


def product_settings_for_simplified_run(
    product_settings: ProductSettings,
    decision: SimplifiedCapabilityDecision,
) -> ProductSettings:
    if decision.capability is SimplifiedSolveCapability.FULL_LOCAL:
        return product_settings
    if decision.capability is SimplifiedSolveCapability.NEAR_ONLY:
        near_mode = str(getattr(product_settings, "near_catalog_mode", "auto") or "auto").strip().lower().replace("_", "-")
        if decision.catalog_source_used != "library" and near_mode == "auto":
            near_mode = "astap-native"
        return replace(product_settings, blind_enabled=False, blind_only=False, near_catalog_mode=near_mode)
    return product_settings
