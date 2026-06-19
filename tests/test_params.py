"""Tests for DetectorParams validation (Task 1) and __repr__ (Task 2)."""

import math

import numpy as np
import pytest

import nanofractal as nf

BLANK = np.full((480, 640), 255, dtype=np.uint8)
DICT = nf.Dict.DICT_4X4_50


# ---------------------------------------------------------------------------
# Task 1 — adaptive_block_size
# ---------------------------------------------------------------------------


def test_adaptive_block_size_even_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.adaptive_block_size = 4
    with pytest.raises(ValueError, match="adaptive_block_size"):
        det.detect(BLANK)


def test_adaptive_block_size_2_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.adaptive_block_size = 2
    with pytest.raises(ValueError, match="adaptive_block_size"):
        det.detect(BLANK)


def test_adaptive_block_size_odd_ok():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.adaptive_block_size = 13
    # Must not raise; blank image returns empty results
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


def test_adaptive_block_size_minus_one_ok():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.adaptive_block_size = -1
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


# ---------------------------------------------------------------------------
# Task 1 — min_contour_size
# ---------------------------------------------------------------------------


def test_min_contour_size_zero_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.min_contour_size = 0
    with pytest.raises(ValueError, match="min_contour_size"):
        det.detect(BLANK)


def test_min_contour_size_minus_one_ok():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.min_contour_size = -1
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


def test_min_contour_size_positive_ok():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.min_contour_size = 30
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


# ---------------------------------------------------------------------------
# Task 1 — approx_poly_rate
# ---------------------------------------------------------------------------


def test_approx_poly_rate_zero_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.approx_poly_rate = 0
    with pytest.raises(ValueError, match="approx_poly_rate"):
        det.detect(BLANK)


def test_approx_poly_rate_negative_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.approx_poly_rate = -0.1
    with pytest.raises(ValueError, match="approx_poly_rate"):
        det.detect(BLANK)


def test_approx_poly_rate_inf_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.approx_poly_rate = math.inf
    with pytest.raises(ValueError, match="approx_poly_rate"):
        det.detect(BLANK)


def test_approx_poly_rate_positive_ok():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.approx_poly_rate = 0.1
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


# ---------------------------------------------------------------------------
# Task 1 — adaptive_c
# ---------------------------------------------------------------------------


def test_adaptive_c_nan_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.adaptive_c = float("nan")
    with pytest.raises(ValueError, match="adaptive_c"):
        det.detect(BLANK)


def test_adaptive_c_inf_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.adaptive_c = float("inf")
    with pytest.raises(ValueError, match="adaptive_c"):
        det.detect(BLANK)


def test_adaptive_c_finite_ok():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.adaptive_c = -5.0
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


# ---------------------------------------------------------------------------
# Task 1 — subpix_win_size
# ---------------------------------------------------------------------------


def test_subpix_win_size_minus_two_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.subpix_win_size = -2
    with pytest.raises(ValueError, match="subpix_win_size"):
        det.detect(BLANK)


def test_subpix_win_size_minus_one_ok():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.subpix_win_size = -1
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


def test_subpix_win_size_zero_ok():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.subpix_win_size = 0
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


# ---------------------------------------------------------------------------
# Task 1 — kfilter_min_dist
# ---------------------------------------------------------------------------


def test_kfilter_min_dist_negative_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.kfilter_min_dist = -1.0
    with pytest.raises(ValueError, match="kfilter_min_dist"):
        det.detect(BLANK)


def test_kfilter_min_dist_inf_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.kfilter_min_dist = float("inf")
    with pytest.raises(ValueError, match="kfilter_min_dist"):
        det.detect(BLANK)


def test_kfilter_min_dist_zero_ok():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.kfilter_min_dist = 0.0
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


# ---------------------------------------------------------------------------
# Task 1 — detection_scale
# ---------------------------------------------------------------------------


def test_detection_scale_nan_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.detection_scale = float("nan")
    with pytest.raises(ValueError, match="detection_scale"):
        det.detect(BLANK)


def test_detection_scale_inf_raises():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.detection_scale = float("inf")
    with pytest.raises(ValueError, match="detection_scale"):
        det.detect(BLANK)


def test_detection_scale_zero_ok():
    """detection_scale <= 0 means no scaling — must not raise."""
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.detection_scale = 0.0
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


def test_detection_scale_negative_ok():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.detection_scale = -1.0
    result = det.detect(BLANK)
    assert result.ids.shape == (0,)


# ---------------------------------------------------------------------------
# Task 1 — validation triggers through detect_batch
# ---------------------------------------------------------------------------


def test_validation_via_detect_batch():
    det = nf.ArucoDetector(DICT, max_attempts=1)
    det.params.adaptive_block_size = 4
    with pytest.raises(ValueError, match="adaptive_block_size"):
        det.detect_batch([BLANK])


# ---------------------------------------------------------------------------
# Task 1 — FractalDetector also validates
# ---------------------------------------------------------------------------


def test_fractal_detect_validates_params():
    det = nf.FractalDetector("FRACTAL_5L_6")
    det.params.adaptive_block_size = 6  # even → invalid
    with pytest.raises(ValueError, match="adaptive_block_size"):
        det.detect(BLANK)


def test_fractal_detect_batch_validates_params():
    det = nf.FractalDetector("FRACTAL_5L_6")
    det.params.min_contour_size = 0
    with pytest.raises(ValueError, match="min_contour_size"):
        det.detect_batch([BLANK])


# ---------------------------------------------------------------------------
# Task 2 — __repr__
# ---------------------------------------------------------------------------


def test_aruco_repr_contains_dict_name():
    det = nf.ArucoDetector(nf.Dict.DICT_4X4_50, max_attempts=3)
    r = repr(det)
    assert "DICT_4X4_50" in r
    assert "3" in r  # max_attempts


def test_aruco_repr_format():
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=1)
    r = repr(det)
    assert "ArucoDetector" in r
    assert "ARUCO_MIP_36h12" in r


def test_fractal_repr_contains_config():
    det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.85)
    r = repr(det)
    assert "FractalDetector" in r
    assert "FRACTAL_5L_6" in r
    assert "0.85" in r


def test_fractal_repr_marker_size():
    det = nf.FractalDetector("FRACTAL_3L_6", marker_size=1.2)
    r = repr(det)
    assert "1.2" in r
