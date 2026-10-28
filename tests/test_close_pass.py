"""Second threshold pass for markers seen from very close (R21)."""

import cv2
import numpy as np
import pytest

import nanofractal as nf

CONFIG = "FRACTAL_3L_6"


def _close_dark(seed, size=2200):
    """A dark, low-contrast 1280x720 frame cut from a marker ``size`` px wide:
    cells of ~160 px, far larger than the automatic threshold block."""
    rng = np.random.default_rng(seed)
    black, white = rng.uniform(4, 10), rng.uniform(22, 40)
    m = nf.generate_fractal(CONFIG, size).astype(np.float32) / 255.0
    img = black + (white - black) * m
    c = size / 2
    dx, dy = rng.uniform(-150, 150), rng.uniform(-100, 100)
    src = np.float32(
        [[c - 640 + dx, c - 360 + dy], [c + 640 + dx, c - 360 + dy],
         [c + 640 + dx, c + 360 + dy], [c - 640 + dx, c + 360 + dy]]
    )  # fmt: skip
    t = rng.uniform(0, 0.08) * 1280
    dst = np.float32([[t, 0], [1280 - t, 0], [1280, 720], [0, 720]])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (1280, 720))
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.5))
    out = out + rng.normal(0, rng.uniform(2, 4), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


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
    img = cv2.GaussianBlur(rng.integers(0, 256, (720, 1280), dtype=np.uint8), (0, 0), 3)
    for _ in range(30):  # dark rectangles and lines
        x, y = rng.integers(0, 1200), rng.integers(0, 650)
        w, h = rng.integers(20, 300), rng.integers(20, 300)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), int(rng.integers(0, 80)), -1)
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
