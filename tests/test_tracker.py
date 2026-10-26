"""FractalTracker: ROI search around the previous detection."""

import numpy as np
import pytest

import nanofractal as nf
from nanofractal.bench import _background, _blur, _warp_onto

CONFIG = "FRACTAL_5L_6"


def _video(w=1280, h=720, n=12):
    rng = np.random.default_rng(0)
    bg = _background(w, h, rng)
    side = 300
    m = np.full((side + 50, side + 50), 255, np.uint8)
    m[25 : 25 + side, 25 : 25 + side] = nf.generate_fractal(CONFIG, side)
    frames = []
    for i in range(n):
        cx, cy, s = 300 + 40 * i, 360 + 10 * np.sin(i), 350 + 15 * i
        quad = np.array(
            [[cx - s / 2, cy - s / 2], [cx + s / 2, cy - s / 2], [cx + s / 2, cy + s / 2],
             [cx - s / 2, cy + s / 2]]
        )
        img = bg.copy()
        _warp_onto(img, m, quad)
        img = _blur(img, 0.8) + rng.normal(0, 3.0, img.shape)
        frames.append(np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8)))
    return frames


def test_tracker_matches_full_frame_detection():
    frames = _video()
    full = nf.FractalDetector(CONFIG)
    tr = nf.FractalTracker(nf.FractalDetector(CONFIG))
    for f in frames:
        a, b = full.detect(f), tr.detect(f)
        assert sorted(a.ids.tolist()) == sorted(b.ids.tolist())
        for mid, c in a:
            np.testing.assert_allclose(b.corners[b.ids.tolist().index(mid)], c, atol=1e-4)
    # the region search did the work on most frames
    assert tr.full_frame_detections <= len(frames) // 2


def test_tracker_inner_points_and_reset():
    frames = _video(n=3)
    tr = nf.FractalTracker(nf.FractalDetector(CONFIG, marker_size=1.0))
    for f in frames:
        res = tr.detect(f, with_inner_points=True)
        assert res.points_2d is not None and len(res) > 0
        assert (res.points_2d[:, 0] >= 0).all() and (res.points_2d[:, 0] < f.shape[1]).all()
    before = tr.full_frame_detections
    tr.reset()
    tr.detect(frames[0])
    assert tr.full_frame_detections == before + 1


def test_tracker_recovers_when_marker_jumps():
    frames = _video(n=2)
    tr = nf.FractalTracker(nf.FractalDetector(CONFIG))
    tr.detect(frames[0])
    jumped = np.ascontiguousarray(np.roll(frames[1], 600, axis=1))  # marker moved far away
    res = tr.detect(jumped)
    assert 0 in res.ids.tolist()


def test_tracker_restores_auto_block_size():
    det = nf.FractalDetector(CONFIG)
    tr = nf.FractalTracker(det)
    for f in _video(n=2):
        tr.detect(f)
    assert det.params.adaptive_block_size == -1


def test_tracker_rejects_bad_margin():
    with pytest.raises(ValueError):
        nf.FractalTracker(nf.FractalDetector(CONFIG), margin=-1)
