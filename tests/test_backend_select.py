# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : MIT (voir pyproject.toml / repository metadata)               ║
# ║                                                                                   ║
# ║ Remerciements amont :                                                             ║
# ║ - ASTAP, par Han Kleijn                                                           ║
# ║ - Astrometry.net, par Dustin Lang, David W. Hogg, Keir Mierle, et al.            ║
# ║                                                                                   ║
# ║ Description FR :                                                                  ║
# ║ Ce code sert à transformer des nuages de photons en solutions WCS et en images   ║
# ║ astronomiques exploitables. Merci de créditer les auteurs et projets amont lors   ║
# ║ de toute réutilisation.                                                           ║
# ║                                                                                   ║
# ║ EN Description:                                                                    ║
# ║ This code helps turn clouds of photons into usable WCS solutions and astronomical ║
# ║ imagery outputs. Please credit both project authors and upstream references when  ║
# ║ reusing this work.                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
# """

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


