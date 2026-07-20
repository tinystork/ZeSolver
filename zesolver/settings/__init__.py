"""P2A settings/profile separation layer."""

from .assembly import ResolvedSolverConfiguration, build_solver_configuration
from .migration import SettingsMigrationResult, migrate_persistent_settings_v2
from .product import ProductSettings, ProfileSelection
from .profiles import (
    BlindSolverProfile,
    NearSolverProfile,
    PipelineProfile,
    get_blind_profile,
    get_near_profile,
    get_pipeline_profile,
)
from .runtime import DeveloperOverrides, RuntimeOptions

__all__ = [
    "BlindSolverProfile",
    "DeveloperOverrides",
    "NearSolverProfile",
    "PipelineProfile",
    "ProductSettings",
    "ProfileSelection",
    "ResolvedSolverConfiguration",
    "RuntimeOptions",
    "SettingsMigrationResult",
    "build_solver_configuration",
    "get_blind_profile",
    "get_near_profile",
    "get_pipeline_profile",
    "migrate_persistent_settings_v2",
]
