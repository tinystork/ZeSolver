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

from pathlib import Path
import hashlib

from zeblindsolver.downloads import DownloadsManager, FakeBackend


def test_download_manager_with_fake_backend(tmp_path: Path):
    content = b"abc123\n" * 10
    sha = hashlib.sha256(content).hexdigest()
    mgr = DownloadsManager(backend=FakeBackend(content))
    item = mgr.add("https://example.invalid/file.bin", tmp_path, filename="file.bin", sha256=sha)

    updates = []

    def on_update(i):
        updates.append((i.status, i.bytes_done, i.bytes_total))

    stop = __import__("threading").Event()
    mgr.run_all(stop_event=stop, on_update=on_update)

    assert item.dest_path.exists()
    assert item.status in {"done", "verified"}
    # Since we provided sha, expect verification to succeed
    assert item.status == "verified"
    assert item.bytes_done == len(content)
    assert updates, "should receive progress updates"

