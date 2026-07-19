from __future__ import annotations

from tests.p3av3_helpers import load_zesolver_app, write_fits


def test_extension_only_wcs_is_not_primary_solved_status(tmp_path) -> None:
    app = load_zesolver_app()
    path = write_fits(tmp_path / "extension_only.fit", primary_wcs=False, extension_wcs=True)

    state = app.inspect_effective_wcs_state(path)

    assert state.status == "waiting"
    assert state.primary_has_wcs is False
    assert state.other_hdus_have_wcs is True
    assert "extension WCS" in state.detail
