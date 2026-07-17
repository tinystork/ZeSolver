from __future__ import annotations

from pathlib import Path

from zesolver.settings import migrate_persistent_settings_v2
from zesolver.settings_store import PersistentSettings


def test_migration_preserves_legacy_paths_as_deprecated_diagnostics(tmp_path: Path) -> None:
    settings = PersistentSettings(
        catalog_library_path=str(tmp_path / "library"),
        db_root=str(tmp_path / "db"),
        index_root=str(tmp_path / "index"),
        blind_4d_manifest_path=str(tmp_path / "manifest.json"),
    )

    result = migrate_persistent_settings_v2(settings)

    assert result.product.catalog_library_path == tmp_path / "library"
    assert "db_root" in result.deprecated
    assert "index_root" in result.deprecated
    assert "blind_4d_manifest_path" in result.deprecated


def test_historical_profile_is_preserved_as_diagnostic_only() -> None:
    settings = PersistentSettings(blind_backend_profile="historical")

    result = migrate_persistent_settings_v2(settings)

    assert result.product.profiles.blind == "zeblind4d-v1"
    assert result.historical_diagnostic_profile == "historical"
    assert "historical_profile_preserved_as_diagnostic" in result.warnings


def test_invalid_settings_migrate_to_safe_values() -> None:
    settings = PersistentSettings(solver_workers=-8, solver_downsample=0, astrometry_timeout_s=2)

    result = migrate_persistent_settings_v2(settings)

    assert result.product.workers == "auto"
    assert result.product.downsample == 1
    assert result.product.astrometry_timeout_s == 30
