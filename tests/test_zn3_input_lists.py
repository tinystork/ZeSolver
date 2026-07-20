from __future__ import annotations

import numpy as np

from zeblindsolver.metadata_solver import (
    astap_background_noise_stats,
    astap_binned_to_full_coords,
    astap_compatible_image_detection,
    astap_compatible_mean_bin_image,
    astap_full_to_binned_coords,
    astap_sqrt_snr_selection_mask,
    choose_astap_compatible_bin_factor,
)


def test_astap_mean_binning_x2_even_dimensions() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    binned, factor = astap_compatible_mean_bin_image(image, 2)

    assert factor == 2
    np.testing.assert_allclose(binned, np.array([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32))


def test_astap_mean_binning_x2_odd_dimensions_truncates() -> None:
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    binned, factor = astap_compatible_mean_bin_image(image, 2)

    assert factor == 2
    assert binned.shape == (2, 2)
    np.testing.assert_allclose(binned, np.array([[3.0, 5.0], [13.0, 15.0]], dtype=np.float32))


def test_astap_binned_to_full_uses_half_pixel_for_x2() -> None:
    x_full, y_full = astap_binned_to_full_coords(10.0, 20.0, 2)

    assert x_full == 20.5
    assert y_full == 40.5


def test_astap_coordinate_round_trip_with_crop_shift() -> None:
    x = np.array([0.0, 10.25, 99.5])
    y = np.array([4.0, 20.5, 50.0])

    xf, yf = astap_binned_to_full_coords(x, y, 3, crop=0.8, width=1000, height=800)
    xb, yb = astap_full_to_binned_coords(xf, yf, 3, crop=0.8, width=1000, height=800)

    np.testing.assert_allclose(xb, x)
    np.testing.assert_allclose(yb, y)


def test_astap_background_noise_stats_deterministic() -> None:
    image = np.array([[10, 10, 10], [10, 20, 10], [10, 10, 10]], dtype=np.float32)

    stats = astap_background_noise_stats(image)

    assert stats["background"] == 10.0
    assert stats["noise"] >= 0.0
    assert stats["threshold_7sigma"] >= stats["background"]


def test_astap_sqrt_snr_selection_preserves_scan_order() -> None:
    scores = np.array([10.0, 100.0, 25.0, 90.0, 80.0])

    mask = astap_sqrt_snr_selection_mask(scores, 3)

    assert mask[1]
    assert mask[3]
    assert mask[4]
    assert not mask[0]
    assert int(mask.sum()) >= 3


def test_astap_compatible_detection_is_deterministic() -> None:
    image = np.zeros((80, 80), dtype=np.float32)
    image[20:23, 20:23] = 500.0
    image[50:53, 48:51] = 800.0

    stars1, diag1 = astap_compatible_image_detection(image, bin_factor=1, max_stars=10, k_sigma=2.0, min_area=2)
    stars2, diag2 = astap_compatible_image_detection(image, bin_factor=1, max_stars=10, k_sigma=2.0, min_area=2)

    assert int(stars1.size) == int(stars2.size)
    assert diag1["detected_count"] == diag2["detected_count"]
    np.testing.assert_allclose(stars1["x"], stars2["x"])
    np.testing.assert_allclose(stars1["y"], stars2["y"])


def test_choose_astap_compatible_bin_factor_policy() -> None:
    assert choose_astap_compatible_bin_factor(width=1080, height=1920, fov_deg=1.27, scale_arcsec=2.39) == 2
    assert choose_astap_compatible_bin_factor(width=300, height=300, fov_deg=0.3, scale_arcsec=4.0) == 1
    assert choose_astap_compatible_bin_factor(width=300, height=300, requested=2) == 2
