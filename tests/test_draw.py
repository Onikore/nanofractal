import numpy as np

import nanofractal as nf
import nanofractal._nanofractal as _nf


def _camera(img):
    h, w = img.shape[:2]
    cam = np.array([[600.0, 0, w / 2.0], [0, 600.0, h / 2.0], [0, 0, 1]], dtype=np.float64)
    return cam, np.zeros(5, dtype=np.float64)


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


def test_fractal_draw_inplace_false_works_on_readonly():
    img = _fractal_bgr()
    det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.85)
    res = det.detect(img)
    img.flags.writeable = False
    out = det.draw(img, res, inplace=False)  # must not raise
    assert out is not img


def test_fractal_draw_with_axes_changes_pixels():
    img = _fractal_bgr()
    det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.85)
    res = det.detect(img)
    cam, dist = _camera(img)
    rvec, tvec, _ = det.estimate_pose(res, cam, dist)
    plain = det.draw(img, res, inplace=False)
    with_axes = det.draw(img, res, cam, dist, rvec, tvec, inplace=False)
    assert not np.array_equal(plain, with_axes)
