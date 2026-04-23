#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ZeSolver icon files (.png/.ico/.icns) from a source image")
    parser.add_argument("--source", default="icon/ZSicon.jpeg", help="Source image path (relative to repo root by default)")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()

    root = Path(args.repo_root).expanduser().resolve()
    source = Path(args.source)
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        print(f"[FAIL] missing source image: {source}")
        return 2

    icon_dir = source.parent
    stem = source.stem

    img = Image.open(source).convert("RGBA")
    size = min(img.size)
    if img.size[0] != img.size[1]:
        left = (img.size[0] - size) // 2
        top = (img.size[1] - size) // 2
        img = img.crop((left, top, left + size, top + size))

    out_png = icon_dir / f"{stem}.png"
    out_ico = icon_dir / f"{stem}.ico"
    out_icns = icon_dir / f"{stem}.icns"

    img.resize((1024, 1024), Image.Resampling.LANCZOS).save(out_png, format="PNG")
    img.save(out_ico, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    img.resize((1024, 1024), Image.Resampling.LANCZOS).save(out_icns, format="ICNS")

    print(f"[OK] {out_png}")
    print(f"[OK] {out_ico}")
    print(f"[OK] {out_icns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
