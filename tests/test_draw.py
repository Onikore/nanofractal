import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf


def _aruco_bgr(marker_id=0, dictionary=int(nf.Dict.DICT_4X4_50), cell=40, margin=60):
    grid = np.asarray(_nf._aruco_marker_image8(dictionary, marker_id))
    up = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    h, w = up.shape
    gray = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    gray[margin : margin + h, margin : margin + w] = up
    return np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))  # (H,W,3) BGR


def _camera(img):
    h, w = img.shape[:2]
    cam = np.array([[600.0, 0, w / 2.0], [0, 600.0, h / 2.0], [0, 0, 1]], dtype=np.float64)
    return cam, np.zeros(5, dtype=np.float64)


# ---------------------------------------------------------------------------
# ArucoDetector.draw must exist (the documented-but-missing method)
# ---------------------------------------------------------------------------


def test_aruco_detector_has_draw():
    assert hasattr(nf.ArucoDetector, "draw")


def test_aruco_draw_outlines_changes_pixels():
    img = _aruco_bgr(7)
    det = nf.ArucoDetector(nf.Dict.DICT_4X4_50, max_attempts=10)
    res = det.detect(img)
    assert len(res) >= 1
    before = img.copy()
    out = det.draw(img, res)
    assert out is img  # inplace default
    assert not np.array_equal(img, before)  # pixels changed
    assert (img[..., 1] == 255).any()  # green outline drawn


def test_aruco_draw_with_per_marker_axes_runs():
    img = _aruco_bgr(7)
    det = nf.ArucoDetector(nf.Dict.DICT_4X4_50, max_attempts=10)
    res = det.detect(img)
    cam, dist = _camera(img)
    rvecs, tvecs = det.estimate_pose(res.corners, cam, dist, marker_size=0.05)
    before = img.copy()
    det.draw(img, res, cam, dist, rvecs, tvecs, marker_size=0.05)
    assert not np.array_equal(img, before)


# ---------------------------------------------------------------------------
# inplace flag — both detectors
# ---------------------------------------------------------------------------


def test_aruco_draw_inplace_false_returns_new_array():
    img = _aruco_bgr(7)
    det = nf.ArucoDetector(nf.Dict.DICT_4X4_50, max_attempts=10)
    res = det.detect(img)
    before = img.copy()
    out = det.draw(img, res, inplace=False)
    assert out is not img  # new array
    assert np.array_equal(img, before)  # original untouched
    assert not np.array_equal(out, before)  # copy was drawn on


def test_aruco_draw_inplace_false_works_on_readonly():
    img = _aruco_bgr(7)
    det = nf.ArucoDetector(nf.Dict.DICT_4X4_50, max_attempts=10)
    res = det.detect(img)
    img.flags.writeable = False
    out = det.draw(img, res, inplace=False)  # must not raise
    assert out is not img


def test_aruco_draw_inplace_true_readonly_raises():
    img = _aruco_bgr(7)
    det = nf.ArucoDetector(nf.Dict.DICT_4X4_50, max_attempts=10)
    res = det.detect(img)
    img.flags.writeable = False
    with pytest.raises(ValueError):
        det.draw(img, res)  # inplace default needs writable


def _fractal_bgr(cell=40, margin=80):
    grid = np.asarray(_nf._fractal_external_image8("FRACTAL_5L_6"))
    up = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    h, w = up.shape
    gray = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    gray[margin : margin + h, margin : margin + w] = up
    return np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))


def test_fractal_draw_inplace_false_returns_new_array():
    img = _fractal_bgr()
    det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.85)
    res = det.detect(img)
    before = img.copy()
    out = det.draw(img, res, inplace=False)
    assert out is not img
    assert np.array_equal(img, before)
    assert not np.array_equal(out, before)


def test_fractal_draw_inplace_true_default_modifies():
    img = _fractal_bgr()
    det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.85)
    res = det.detect(img)
    before = img.copy()
    out = det.draw(img, res)
    assert out is img
    assert not np.array_equal(img, before)
