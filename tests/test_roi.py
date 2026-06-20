"""TDD: B1 — ROI detection for ArucoDetector and FractalDetector."""

import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf

CONFIG = "FRACTAL_5L_6"


# ── helpers ──────────────────────────────────────────────────────────────────


def _upscale(grid8: np.ndarray, cell: int) -> np.ndarray:
    return np.kron(grid8, np.ones((cell, cell), dtype=np.uint8))


def _render_aruco_in_canvas(
    marker_id: int = 0,
    dictionary: int = 0,
    cell: int = 40,
    margin: int = 60,
):
    """Return (canvas uint8 H×W, x_offset, y_offset) where marker starts."""
    grid = np.asarray(_nf._aruco_marker_image8(dictionary, marker_id))
    marker = _upscale(grid, cell)
    h, w = marker.shape
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = marker
    return np.ascontiguousarray(canvas), margin, margin


def _render_fractal_in_canvas(cell: int = 40, margin: int = 80):
    grid = np.asarray(_nf._fractal_external_image8(CONFIG))
    up = _upscale(grid, cell)
    h, w = up.shape
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = up
    return np.ascontiguousarray(canvas), margin, margin


# ── ArucoDetector.detect ─────────────────────────────────────────────────────


def test_aruco_roi_detect_finds_marker():
    img, ox, oy = _render_aruco_in_canvas(marker_id=0)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    H, W = img.shape
    # ROI with 10px padding around the marker
    roi = (ox - 10, oy - 10, W - 2 * (ox - 10), H - 2 * (oy - 10))
    res = det.detect(img, roi=roi)
    assert 0 in res.ids.tolist()


def test_aruco_roi_corners_in_full_image_coords():
    """Corners returned through an ROI must be in full-image space."""
    img, ox, oy = _render_aruco_in_canvas(marker_id=0, margin=80)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    H, W = img.shape
    rx, ry = ox - 20, oy - 20
    rw, rh = W - 2 * (ox - 20), H - 2 * (oy - 20)
    res_roi = det.detect(img, roi=(rx, ry, rw, rh))
    res_full = det.detect(img)
    assert 0 in res_roi.ids.tolist(), "marker not found with ROI"
    assert 0 in res_full.ids.tolist(), "marker not found without ROI"
    c_roi = res_roi.corners[res_roi.ids.tolist().index(0)]
    c_full = res_full.corners[res_full.ids.tolist().index(0)]
    # If coords were ROI-local they'd be ~70 px smaller — a 5 px tolerance is
    # tight enough to distinguish that.
    np.testing.assert_allclose(c_roi, c_full, atol=5.0)


def test_aruco_roi_none_matches_no_roi():
    img, _, _ = _render_aruco_in_canvas(marker_id=1)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res_none = det.detect(img, roi=None)
    res_noroi = det.detect(img)
    assert res_none.ids.tolist() == res_noroi.ids.tolist()
    np.testing.assert_allclose(res_none.corners, res_noroi.corners, atol=1e-4)


def test_aruco_roi_out_of_bounds_raises():
    img, _, _ = _render_aruco_in_canvas()
    det = nf.ArucoDetector()
    H, W = img.shape
    with pytest.raises(ValueError, match="roi"):
        det.detect(img, roi=(0, 0, W + 1, H))


def test_aruco_roi_nonpositive_width_raises():
    img, _, _ = _render_aruco_in_canvas()
    det = nf.ArucoDetector()
    with pytest.raises(ValueError):
        det.detect(img, roi=(0, 0, 0, 100))


def test_aruco_roi_nonpositive_height_raises():
    img, _, _ = _render_aruco_in_canvas()
    det = nf.ArucoDetector()
    with pytest.raises(ValueError):
        det.detect(img, roi=(0, 0, 100, -1))


def test_aruco_roi_negative_x_raises():
    img, _, _ = _render_aruco_in_canvas()
    det = nf.ArucoDetector()
    with pytest.raises(ValueError):
        det.detect(img, roi=(-1, 0, 100, 100))


def test_aruco_roi_bad_length_raises():
    img, _, _ = _render_aruco_in_canvas()
    det = nf.ArucoDetector()
    with pytest.raises(ValueError):
        det.detect(img, roi=(0, 0, 100))  # only 3 elements


# ── ArucoDetector.detect_batch ───────────────────────────────────────────────


def test_aruco_detect_batch_roi_finds_marker():
    img, ox, oy = _render_aruco_in_canvas(marker_id=0)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    H, W = img.shape
    roi = (ox - 10, oy - 10, W - 2 * (ox - 10), H - 2 * (oy - 10))
    results = det.detect_batch([img, img], roi=roi)
    assert len(results) == 2
    for res in results:
        assert 0 in res.ids.tolist()


def test_aruco_detect_batch_roi_corners_in_full_image_coords():
    img, ox, oy = _render_aruco_in_canvas(marker_id=0, margin=80)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    H, W = img.shape
    rx, ry = ox - 20, oy - 20
    roi = (rx, ry, W - 2 * (ox - 20), H - 2 * (oy - 20))
    results_roi = det.detect_batch([img], roi=roi)
    results_full = det.detect_batch([img])
    c_roi = results_roi[0].corners[results_roi[0].ids.tolist().index(0)]
    c_full = results_full[0].corners[results_full[0].ids.tolist().index(0)]
    np.testing.assert_allclose(c_roi, c_full, atol=5.0)


def test_aruco_detect_batch_roi_validates_each_image():
    """ROI that overflows a differently-sized image in the batch raises."""
    img_big, _, _ = _render_aruco_in_canvas(margin=80)  # 480×480
    img_small = img_big[:200, :200]  # 200×200
    det = nf.ArucoDetector()
    H, W = img_big.shape
    # roi fits img_big but not img_small
    roi = (0, 0, W, H)
    with pytest.raises(ValueError):
        det.detect_batch([img_big, img_small], roi=roi)


# ── FractalDetector.detect ───────────────────────────────────────────────────


def test_fractal_roi_detect_finds_marker():
    img, ox, oy = _render_fractal_in_canvas()
    det = nf.FractalDetector(CONFIG)
    H, W = img.shape
    roi = (ox - 20, oy - 20, W - 2 * (ox - 20), H - 2 * (oy - 20))
    res = det.detect(img, roi=roi)
    ext_id = _nf._fractal_external_id(CONFIG)
    assert ext_id in res.ids.tolist()


def test_fractal_roi_corners_in_full_image_coords():
    img, ox, oy = _render_fractal_in_canvas()
    det = nf.FractalDetector(CONFIG)
    H, W = img.shape
    rx, ry = ox - 20, oy - 20
    roi = (rx, ry, W - 2 * (ox - 20), H - 2 * (oy - 20))
    res_roi = det.detect(img, roi=roi)
    res_full = det.detect(img)
    ext_id = _nf._fractal_external_id(CONFIG)
    assert ext_id in res_roi.ids.tolist()
    assert ext_id in res_full.ids.tolist()
    c_roi = res_roi.corners[res_roi.ids.tolist().index(ext_id)]
    c_full = res_full.corners[res_full.ids.tolist().index(ext_id)]
    np.testing.assert_allclose(c_roi, c_full, atol=5.0)


def test_fractal_roi_none_matches_no_roi():
    img, _, _ = _render_fractal_in_canvas()
    det = nf.FractalDetector(CONFIG)
    res_none = det.detect(img, roi=None)
    res_noroi = det.detect(img)
    assert res_none.ids.tolist() == res_noroi.ids.tolist()


# ── FractalDetector.detect_batch ─────────────────────────────────────────────


def test_fractal_detect_batch_roi_finds_marker():
    img, ox, oy = _render_fractal_in_canvas()
    det = nf.FractalDetector(CONFIG)
    H, W = img.shape
    roi = (ox - 20, oy - 20, W - 2 * (ox - 20), H - 2 * (oy - 20))
    results = det.detect_batch([img, img], roi=roi)
    ext_id = _nf._fractal_external_id(CONFIG)
    assert len(results) == 2
    for res in results:
        assert ext_id in res.ids.tolist()


# ── ROI boundary cases ───────────────────────────────────────────────────────


def test_aruco_roi_full_image_matches_no_roi():
    """roi covering the whole image gives the same ids as no roi at all."""
    img, _, _ = _render_aruco_in_canvas(marker_id=0)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    H, W = img.shape
    res_full_roi = det.detect(img, roi=(0, 0, W, H))
    res_no_roi = det.detect(img)
    assert sorted(res_full_roi.ids.tolist()) == sorted(res_no_roi.ids.tolist())


def test_aruco_roi_excludes_marker_returns_empty():
    """roi that does not cover the marker should return no detections."""
    img, ox, oy = _render_aruco_in_canvas(marker_id=0, margin=80)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    H, W = img.shape
    # Marker sits at top-left (starting at ox, oy); roi covers bottom-right quadrant only.
    roi = (W // 2, H // 2, W - W // 2, H - H // 2)
    res = det.detect(img, roi=roi)
    assert 0 not in res.ids.tolist()


def test_aruco_roi_flush_right_bottom_accepted():
    """roi with x+w == W and y+h == H (flush edges) must not raise an off-by-one error."""
    img, _, _ = _render_aruco_in_canvas(marker_id=0)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    H, W = img.shape
    # Inset by 1 on top/left so the roi is *not* the full frame, but still flush right/bottom.
    roi = (1, 1, W - 1, H - 1)
    assert roi[0] + roi[2] == W  # x + w == W
    assert roi[1] + roi[3] == H  # y + h == H
    # Must not raise; marker is well inside so we can also assert detection.
    res = det.detect(img, roi=roi)
    assert 0 in res.ids.tolist()
