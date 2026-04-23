# PyInstaller packaging (ZeSolver)

This folder contains a cross-platform build helper that keeps GUI icons aligned with runtime:

- Windows app icon: `icon/ZSicon.ico`
- macOS app icon: `icon/ZSicon.icns`
- Linux app icon fallback: `icon/ZSicon.png`

## Prerequisites

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
pip install pyinstaller
```

## Regenerate icon files from a source image

```bash
.venv/bin/python packaging/pyinstaller/convert_icon.py --source icon/ZSicon.jpeg
```

## Build (onedir, default)

```bash
.venv/bin/python packaging/pyinstaller/build.py
```

On Windows PowerShell:

```powershell
.venv\Scripts\python.exe packaging\pyinstaller\build.py
```

## Build (onefile)

```bash
.venv/bin/python packaging/pyinstaller/build.py --onefile
```

## Clean build folders first

```bash
.venv/bin/python packaging/pyinstaller/build.py --clean
```

The generated artifacts are under `dist/`.
