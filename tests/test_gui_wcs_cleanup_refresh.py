from __future__ import annotations

import warnings

from zewcscleaner import process_fits

from tests.p3av3_helpers import load_zesolver_app, write_fits


def test_effective_wcs_refresh_after_primary_cleanup(tmp_path) -> None:
    app = load_zesolver_app()
    path = write_fits(tmp_path / "with_wcs.fit", primary_wcs=True)

    before = app.inspect_effective_wcs_state(path)
    assert before.status == "wcs"
    assert before.primary_has_wcs is True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*datetime.datetime.utcnow.*", category=DeprecationWarning)
        deleted, edited_hdus = process_fits(str(path), dry_run=False, backup=False, only_if_wcs=True, all_hdus=False)

    after = app.inspect_effective_wcs_state(path)
    assert deleted > 0
    assert edited_hdus == 1
    assert after.status == "waiting"
    assert after.primary_has_wcs is False
