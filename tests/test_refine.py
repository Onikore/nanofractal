"""TDD: B2 — nf.refine_corners."""

import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf


def _upscale(grid8: np.ndarray, cell: int) -> np.ndarray:
    return np.kron(grid8, np.ones((cell, cell), dtype=np.uint8))


def _render_aruco(marker_id: int = 0, dictionary: int = 0, cell: int = 40, margin: int = 60):
    grid = np.asarray(_nf._aruco_marker_image8(dictionary, marker_id))
    marker = _upscale(grid, cell)
    h, w = marker.shape
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = marker
    return np.ascontiguousarray(canvas)


# ── shape / dtype ─────────────────────────────────────────────────────────────


def test_refine_output_shape_dtype():
    img = _render_aruco(0)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    assert len(res) > 0
    refined = nf.refine_corners(img, res.corners)
    assert refined.shape == res.corners.shape  # (N, 4, 2)
    assert refined.dtype == np.float32


def test_refine_returns_new_array():
    """refine_corners should not mutate the input corners array."""
    img = _render_aruco(0)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    corners_copy = res.corners.copy()
    nf.refine_corners(img, res.corners)
    np.testing.assert_array_equal(res.corners, corners_copy)


# ── sanity: refined corners stay close to detected ───────────────────────────


def test_refine_corners_close_to_detected():
    img = _render_aruco(0)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    assert len(res) > 0
    refined = nf.refine_corners(img, res.corners)
    # Synthetic clean image: subpix refinement moves corners only a little.
    assert np.max(np.abs(refined - res.corners)) < 5.0


# ── win_size validation ───────────────────────────────────────────────────────


def test_refine_win_size_zero_raises():
    img = _render_aruco(0)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    with pytest.raises(ValueError, match="win_size"):
        nf.refine_corners(img, res.corners, win_size=0)


def test_refine_win_size_negative_raises():
    img = _render_aruco(0)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    with pytest.raises(ValueError, match="win_size"):
        nf.refine_corners(img, res.corners, win_size=-3)


# ── BGR (3-channel) input ─────────────────────────────────────────────────────


def test_refine_bgr_image():
    gray = _render_aruco(0)
    bgr = np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(gray)
    assert len(res) > 0
    refined = nf.refine_corners(bgr, res.corners)
    assert refined.shape == res.corners.shape
    assert refined.dtype == np.float32
    # Should produce very similar results to grayscale refinement.
    refined_gray = nf.refine_corners(gray, res.corners)
    np.testing.assert_allclose(refined, refined_gray, atol=0.5)


# ── empty corners ─────────────────────────────────────────────────────────────


def test_refine_empty_corners():
    img = _render_aruco(0)
    empty = np.zeros((0, 4, 2), dtype=np.float32)
    refined = nf.refine_corners(img, empty)
    assert refined.shape == (0, 4, 2)
    assert refined.dtype == np.float32


# ── custom win_size ───────────────────────────────────────────────────────────


def test_refine_custom_win_size():
    img = _render_aruco(0)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    refined = nf.refine_corners(img, res.corners, win_size=3)
    assert refined.shape == res.corners.shape
    assert refined.dtype == np.float32
