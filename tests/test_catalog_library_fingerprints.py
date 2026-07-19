from __future__ import annotations

from zesolver.catalog_library import canonical_provenance_fingerprint


def test_canonical_provenance_fingerprint_ignores_generated_timestamps() -> None:
    base = {
        "engine": "blind4d",
        "source_tiles": ["d50_2823"],
        "parameters": {
            "level": "S",
            "generated_at": "2026-07-19T00:00:00Z",
        },
    }
    changed = {
        "parameters": {
            "generated_at": "2026-07-20T00:00:00Z",
            "level": "S",
        },
        "source_tiles": ["d50_2823"],
        "engine": "blind4d",
    }

    assert canonical_provenance_fingerprint(base) == canonical_provenance_fingerprint(changed)


def test_canonical_provenance_fingerprint_changes_on_scientific_parameter() -> None:
    first = {"engine": "blind4d", "parameters": {"max_stars_per_tile": 2000}}
    second = {"engine": "blind4d", "parameters": {"max_stars_per_tile": 4000}}

    assert canonical_provenance_fingerprint(first) != canonical_provenance_fingerprint(second)
