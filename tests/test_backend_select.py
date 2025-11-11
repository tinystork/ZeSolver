import types
from pathlib import Path


def test_cli_backend_astrometry_selection(monkeypatch, tmp_path):
    # Create a dummy FITS file path (contents not read by the stub)
    img = tmp_path / "image.fits"
    img.write_bytes(b"SIMPLE =                    T\nEND\n")

    # Stub astrometry backend single-file solve
    called = {}

    def fake_solve_single(path, cfg, *, log=None):
        called["path"] = Path(path)
        # Return minimal object with 'success' and 'message'
        return types.SimpleNamespace(success=True, message="ok")

    monkeypatch.setenv("ASTROMETRY_API_KEY", "env-key")
    import zeblindsolver.astrometry_backend as ab
    monkeypatch.setattr(ab, "solve_single", fake_solve_single, raising=True)

    from zeblindsolver.zeblindsolver import main as cli

    rc = cli([
        str(img),
        "--solver-backend", "astrometry",
        "--astrometry-api-url", "http://example/api",
        "--astrometry-api-key", "dummy",
    ])
    assert rc == 0
    assert called.get("path") == img


