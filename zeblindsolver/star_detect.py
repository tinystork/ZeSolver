from __future__ import annotations

import logging
import numpy as np
from scipy.ndimage import gaussian_filter, label

# Optional CUDA backend via CuPy (NVIDIA). Imported lazily in GPU path.
_HAVE_CUPY = False
try:  # pragma: no cover - optional dependency
    import cupy as _cp  # type: ignore
    import cupyx.scipy.ndimage as _cpx_nd  # type: ignore
    from cupy.cuda import runtime as _cuda_rt  # type: ignore
    _HAVE_CUPY = True
except Exception:  # pragma: no cover
    _HAVE_CUPY = False


def _grid_edges(length: int, divisions: int) -> np.ndarray:
    """Return monotonic integer edges covering [0, length]."""
    divisions = max(1, int(divisions))
    edges = np.linspace(0, int(length), divisions + 1, dtype=int)
    edges[0] = 0
    edges[-1] = int(length)
    return edges


def _local_grid_threshold(array, *, k_sigma: float, divisions: int, xp):
    """Compute a per-pixel threshold map using block-wise statistics."""
    divisions = max(1, int(divisions))
    mean = xp.nanmean(array)
    std = xp.nanstd(array)
    threshold = xp.full_like(array, mean + k_sigma * std)
    rows = _grid_edges(array.shape[0], divisions)
    cols = _grid_edges(array.shape[1], divisions)
    for r in range(divisions):
        y0, y1 = int(rows[r]), int(rows[r + 1])
        if y1 <= y0:
            continue
        for c in range(divisions):
            x0, x1 = int(cols[c]), int(cols[c + 1])
            if x1 <= x0:
                continue
            block = array[y0:y1, x0:x1]
            if block.size == 0:
                continue
            local_mean = xp.nanmean(block)
            local_std = xp.nanstd(block)
            threshold[y0:y1, x0:x1] = local_mean + k_sigma * local_std
    return threshold


def detect_stars(
    img: np.ndarray,
    *,
    min_fwhm_px: float = 1.5,
    max_fwhm_px: float = 8.0,
    k_sigma: float = 3.0,
    min_area: int = 5,
    mode: str = "adaptive",  # "adaptive" | "global"
    bg_sigma: float | None = None,
    backend: str = "auto",  # "auto" | "cpu" | "cuda"
    device: int | None = None,
    grid_divisions: int = 8,
    max_labels: int = 5000,
    backend_trace: dict[str, str] | None = None,
) -> np.ndarray:
    """Detect bright, stellar-like sources.

    backend:
      - "auto": prefer CUDA if available, otherwise CPU
      - "cpu": force NumPy/SciPy path
      - "cuda": use CuPy on the selected CUDA device
    device: CUDA device index (0..N-1) when backend="cuda" or "auto" with GPUs.
    mode:
      - "adaptive" (default): divide the smoothed image into a grid, compute local mean/std,
        and threshold each cell independently (robust to gradients)
      - "global": legacy mean + k_sigma threshold
    """
    data = np.asarray(img, dtype=np.float32)
    bg_sigma_val = float(bg_sigma) if bg_sigma is not None else max(min_fwhm_px * 3.0, 5.0)
    use_cuda = False
    if backend.lower() == "cuda":
        use_cuda = _HAVE_CUPY
    elif backend.lower() == "auto":
        use_cuda = _HAVE_CUPY
        if use_cuda:
            try:
                ndev = int(_cuda_rt.getDeviceCount())
                use_cuda = ndev > 0
            except Exception:  # pragma: no cover
                use_cuda = False
    # GPU path (CuPy): accelerate blur/threshold/label/COM via CUDA if available
    if use_cuda:
        try:
            dev_id = int(device) if device is not None else 0
            with _cp.cuda.Device(dev_id):
                d_data = _cp.asarray(data)
                d_blur = _cpx_nd.gaussian_filter(d_data, sigma=min_fwhm_px)
                mode_lower = (mode or "adaptive").lower()
                if mode_lower == "adaptive":
                    d_bg = _cpx_nd.gaussian_filter(d_data, sigma=bg_sigma_val)
                    d_threshold = _local_grid_threshold(
                        d_bg,
                        k_sigma=k_sigma,
                        divisions=grid_divisions,
                        xp=_cp,
                    )
                    d_mask = d_blur > d_threshold
                else:
                    mean = float(_cp.nanmean(d_blur).get())
                    std = float(_cp.nanstd(d_blur).get())
                    threshold = mean + k_sigma * std
                    d_mask = d_blur > threshold
                if not bool(d_mask.any().get()):
                    if backend_trace is not None:
                        backend_trace["used"] = "cuda"
                    return np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
                d_labeled, count = _cpx_nd.label(d_mask)
                count = int(count)
                if count <= 0:
                    if backend_trace is not None:
                        backend_trace["used"] = "cuda"
                    return np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
                flat_labels = d_labeled.ravel()
                flat_input = d_data.ravel()
                # bincount to compute area and flux per label
                areas = _cp.bincount(flat_labels, minlength=count + 1)
                fluxes = _cp.bincount(flat_labels, weights=flat_input, minlength=count + 1)
                # centers of mass for all labels at once
                indices = _cp.arange(1, count + 1, dtype=_cp.int32)
                centers = _cpx_nd.center_of_mass(d_data, d_labeled, indices)
                # Move to host and build output
                areas_h = areas.get()
                fluxes_h = fluxes.get()
                centers_h = [(float(y), float(x)) for (y, x) in centers]
                out = []
                for idx in range(1, count + 1):
                    area = int(areas_h[idx])
                    if area < min_area:
                        continue
                    cy, cx = centers_h[idx - 1]
                    if not (np.isfinite(cx) and np.isfinite(cy)):
                        continue
                    flux = float(fluxes_h[idx])
                    fwhm = min(max_fwhm_px, max(min_fwhm_px, float(np.sqrt(max(1, area)))))
                    out.append((cx, cy, flux, fwhm))
                if not out:
                    if backend_trace is not None:
                        backend_trace["used"] = "cuda"
                    return np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
                arr = np.array(out, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
                order = np.argsort(arr["flux"])[::-1]
                if backend_trace is not None:
                    backend_trace["used"] = "cuda"
                return arr[order]
        except Exception as exc:  # Fallback to CPU on any GPU error
            logging.debug("detect_stars CUDA fallback to CPU: %s", exc)
            if backend_trace is not None:
                backend_trace["fallback"] = "cuda_to_cpu"
                backend_trace["error"] = str(exc)
            pass
    # CPU path (default)
    blurred = gaussian_filter(data, sigma=min_fwhm_px)
    mode_lower = (mode or "adaptive").lower()
    if mode_lower == "adaptive":
        bg = gaussian_filter(data, sigma=bg_sigma_val)
        threshold = _local_grid_threshold(
            bg,
            k_sigma=k_sigma,
            divisions=grid_divisions,
            xp=np,
        )
    else:
        mean = float(np.nanmean(blurred))
        std = float(np.nanstd(blurred))
        threshold = mean + k_sigma * std
    mask = blurred > threshold
    if not mask.any():
        if backend_trace is not None:
            backend_trace["used"] = "cpu"
        return np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
    structure = np.ones((3, 3), dtype=bool)
    labeled, count = label(mask, structure=structure)
    stars = []
    if int(count) > 0:
        flat_labels = labeled.ravel()
        areas = np.bincount(flat_labels, minlength=int(count) + 1)
        candidate_labels = np.flatnonzero(areas >= int(min_area))
        candidate_labels = candidate_labels[candidate_labels > 0]
        cap = max(1, int(max_labels))
        if candidate_labels.size > cap:
            order = np.argsort(areas[candidate_labels])[::-1]
            candidate_labels = candidate_labels[order[:cap]]
        if candidate_labels.size > 0:
            keep = np.zeros(int(count) + 1, dtype=bool)
            keep[candidate_labels] = True
            selected = keep[flat_labels]
            if np.any(selected):
                flat_vals = data.ravel().astype(np.float64, copy=False)
                sel_labels = flat_labels[selected].astype(np.int32, copy=False)
                sel_vals = flat_vals[selected]
                # Pixel coordinates for selected entries in flattened layout.
                flat_idx = np.flatnonzero(selected)
                ys = (flat_idx // data.shape[1]).astype(np.float64, copy=False)
                xs = (flat_idx % data.shape[1]).astype(np.float64, copy=False)
                fluxes = np.bincount(sel_labels, weights=sel_vals, minlength=int(count) + 1)
                sum_y = np.bincount(sel_labels, weights=(sel_vals * ys), minlength=int(count) + 1)
                sum_x = np.bincount(sel_labels, weights=(sel_vals * xs), minlength=int(count) + 1)
                for label_idx in candidate_labels:
                    area = int(areas[int(label_idx)])
                    flux = float(fluxes[int(label_idx)])
                    # Guard against zero/negative total weight.
                    if not np.isfinite(flux) or flux <= 0.0:
                        continue
                    cy = float(sum_y[int(label_idx)] / flux)
                    cx = float(sum_x[int(label_idx)] / flux)
                    if not np.isfinite(cx) or not np.isfinite(cy):
                        continue
                    fwhm = min(max_fwhm_px, max(min_fwhm_px, float(np.sqrt(max(1, area)))))
                    stars.append((cx, cy, flux, fwhm))
    if not stars:
        if backend_trace is not None:
            backend_trace["used"] = "cpu"
        return np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
    array = np.array(stars, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
    order = np.argsort(array["flux"])[::-1]
    if backend_trace is not None:
        backend_trace["used"] = "cpu"
    return array[order]
