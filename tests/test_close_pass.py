"""Second threshold pass for markers seen from very close (R21)."""

import numpy as np
import pytest

import nanofractal as nf

CONFIG = "FRACTAL_3L_6"


def _blur(img, sigma):
    """Separable Gaussian blur (numpy only)."""
    r = int(3 * sigma + 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(img, r, mode="edge")
    p = sum(k[i] * p[:, i : i + img.shape[1]] for i in range(2 * r + 1))
    return sum(k[i] * p[i : i + img.shape[0], :] for i in range(2 * r + 1))


def _close_dark(seed, size=2200):
    """A dark, low-contrast 1280x720 crop of a marker ``size`` px wide:
    cells of ~160 px, far larger than the automatic threshold block."""
    rng = np.random.default_rng(seed)
    black, white = rng.uniform(4, 10), rng.uniform(22, 40)
    m = nf.generate_fractal(CONFIG, size).astype(np.float64) / 255.0
    x0 = int(size / 2 - 640 + rng.uniform(-150, 150))
    y0 = int(size / 2 - 360 + rng.uniform(-100, 100))
    img = black + (white - black) * m[y0 : y0 + 720, x0 : x0 + 1280]
    img = _blur(img, rng.uniform(0.6, 1.5)) + rng.normal(0, rng.uniform(2, 4), img.shape)
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


FRAMES = [_close_dark(s) for s in range(8)]


def test_close_marker_found_only_with_second_pass():
    det = nf.FractalDetector(CONFIG)
    assert det.params.close_pass is True
    found = sum(len(det.detect(f)) > 0 for f in FRAMES)
    det.params.close_pass = False
    found_off = sum(len(det.detect(f)) > 0 for f in FRAMES)
    assert found_off == 0
    assert found >= 7


def test_second_pass_corners_are_refined_at_full_resolution():
    det = nf.FractalDetector(CONFIG)
    res = det.detect(FRAMES[0])
    assert len(res)
    # corners lie on the strong edges of the full frame: cornerSubPix again
    # moves them by far less than the 4x shrink of the second pass
    again = nf.refine_corners(FRAMES[0], res.corners, win_size=4)
    assert np.abs(again - res.corners).max() < 1.0


def test_frames_with_a_marker_are_unchanged():
    page = np.full((900, 1400), 255, np.uint8)
    page[100:800, 350:1050] = nf.generate_fractal(CONFIG, 700)
    on = nf.FractalDetector(CONFIG)
    off = nf.FractalDetector(CONFIG)
    off.params.close_pass = False
    a, b = on.detect(page), off.detect(page)
    assert len(a) > 0
    np.testing.assert_array_equal(a.ids, b.ids)
    np.testing.assert_array_equal(a.corners, b.corners)


@pytest.mark.parametrize("seed", range(5))
def test_no_detection_on_clutter(seed):
    rng = np.random.default_rng(100 + seed)
    img = _blur(rng.uniform(0, 255, (720, 1280)), 3.0)
    for _ in range(30):  # dark rectangles
        x, y = rng.integers(0, 1200), rng.integers(0, 650)
        w, h = rng.integers(20, 300), rng.integers(20, 300)
        img[y : y + h, x : x + w] = rng.uniform(0, 80)
    img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
    assert len(nf.FractalDetector(CONFIG).detect(img)) == 0


def test_roi_and_diagnose():
    det = nf.FractalDetector(CONFIG)
    full = det.detect(FRAMES[1])
    assert len(full)
    roi = det.detect(FRAMES[1], roi=(0, 0, 1280, 720))
    np.testing.assert_array_equal(full.ids, roi.ids)
    d = det.diagnose(FRAMES[1])
    assert d.second_pass and d.result.ids.tolist() == full.ids.tolist()
    assert any(q.reason == "accepted" for q in d.quads)
    # nothing found by either pass: the quads are those of the first pass
    blank = np.full((720, 1280), 128, np.uint8)
    d = det.diagnose(blank)
    assert not d.second_pass and len(d.result) == 0
    det.params.close_pass = False
    assert not det.diagnose(FRAMES[1]).second_pass


def test_params_repr_and_tiny_frames():
    p = nf.DetectorParams()
    assert "close_pass=True" in repr(p)
    det = nf.FractalDetector(CONFIG)
    for shape in ((3, 3), (16, 400), (400, 16), (40, 40)):
        assert len(det.detect(np.zeros(shape, np.uint8))) == 0
