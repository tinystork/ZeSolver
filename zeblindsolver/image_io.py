from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import rawpy  # type: ignore
except Exception:  # pragma: no cover
    rawpy = None  # type: ignore[assignment]

try:  # pragma: no cover
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:  # pragma: no cover
    import imageio.v3 as iio  # type: ignore
except Exception:  # pragma: no cover
    iio = None  # type: ignore[assignment]

RAW_EXTENSIONS = {
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".raf",
    ".dng",
    ".orf",
    ".rw2",
}


def _normalize_to_grayscale(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 3:
        data = np.mean(data, axis=2)
    return data.astype(np.float32, copy=False)


def _normalize_dynamic_range(arr: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(arr, copy=False)
    min_val = float(np.min(data))
    data = data - min_val
    max_val = float(np.max(data))
    if max_val > 0:
        data /= max_val
    return data.astype(np.float32, copy=False)


def _load_with_pil(path: Path) -> tuple[np.ndarray, str]:
    if Image is None:
        raise RuntimeError("Pillow is not installed; install pillow to read raster images")
    with Image.open(path) as img:
        luma = img.convert("L")
        arr = np.array(luma, dtype=np.float32)
    return arr, "PIL"


def _load_with_imageio(path: Path) -> tuple[np.ndarray, str]:
    if iio is None:
        raise RuntimeError("imageio is not installed; install imageio to read raster images")
    arr = iio.imread(path)
    return arr.astype(np.float32, copy=False), "imageio"


def _load_with_rawpy(path: Path) -> tuple[np.ndarray, str]:
    if rawpy is None:
        raise RuntimeError("rawpy is not installed; install rawpy to read RAW images")
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
    return rgb.astype(np.float32, copy=False), "rawpy"


def load_raster_image(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a non-FITS raster image and return a float32 grayscale array."""
    source = Path(path)
    ext = source.suffix.lower()
    metadata: dict[str, Any] = {"source": str(source)}
    if ext in RAW_EXTENSIONS:
        arr, backend = _load_with_rawpy(source)
    else:
        try:
            arr, backend = _load_with_pil(source)
        except Exception:
            arr, backend = _load_with_imageio(source)
    metadata["backend"] = backend
    gray = _normalize_to_grayscale(arr)
    return _normalize_dynamic_range(gray), metadata
