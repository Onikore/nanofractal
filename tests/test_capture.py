"""Capture: FractalTracker(capture_frames=N) picks up a marker before it is read."""

import numpy as np
import pytest

import nanofractal as nf

CONFIG = "FRACTAL_3L_6"
CELL = 50


def _small(flips=(), shift=0):
    """The marker at 5 px per cell (inner levels too small to find), outer-grid
    cells (row, col) inverted so the plain reading fails."""
    m = nf.generate_fractal(CONFIG, 14 * CELL).astype(np.float64)
    for r, c in flips:
        blk = m[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL]
        blk[...] = 255 - blk
    m = m.reshape(70, 10, 70, 10).mean((1, 3))
    page = np.full((240, 320), 255.0)
    page[60 + shift : 130 + shift, 100 + shift : 170 + shift] = m
    return page.astype(np.uint8)


def _captured(det, img):
    det._d.set_capture(True)
    try:
        res = det.detect(img)
    finally:
        det._d.set_capture(False)
    return res, det._d.captured()


@pytest.mark.parametrize("n", [2, 8])
def test_candidates_within_the_prior_tolerance(n):
    det = nf.FractalDetector(CONFIG)
    res, (ids, corners, err) = _captured(det, _small([(2, 2 + k) for k in range(n)]))
    assert len(res) == 0
    assert 0 in ids.tolist() and err[0] == n
    assert corners.shape[1:] == (4, 2)
    np.testing.assert_allclose(corners[0].mean(0), [135, 95], atol=2)


def test_no_candidate_beyond_tolerance_or_when_off():
    det = nf.FractalDetector(CONFIG)
    _, (ids, _, _) = _captured(det, _small([(2, 2 + k) for k in range(9)]))
    assert len(ids) == 0
    det.detect(_small([(2, 3)]))  # capture off: nothing listed
    assert len(det._d.captured()[0]) == 0


def test_no_candidate_when_something_is_detected():
    det = nf.FractalDetector(CONFIG)
    res, (ids, _, _) = _captured(det, _small())
    assert 0 in res.ids and len(ids) == 0


def test_tracker_reads_the_n_th_frame():
    img = _small([(2, 3), (2, 5)])
    det = nf.FractalDetector(CONFIG)
    assert len(det.detect(img)) == 0
    t = nf.FractalTracker(det, capture_frames=3)
    got = [t.detect(img) for _ in range(4)]
    assert [len(r) for r in got[:2]] == [0, 0]
    assert got[2].ids.tolist() == [0] and got[2].bit_errors.tolist() == [2]
    assert got[3].ids.tolist() == [0]  # tracked on
    # plain detect is unchanged by a tracker's capture
    assert len(det.detect(img)) == 0


def test_tracker_needs_the_same_place():
    t = nf.FractalTracker(nf.FractalDetector(CONFIG), capture_frames=3)
    frames = [_small([(2, 3), (2, 5)], shift=s) for s in (0, 90, 0, 90, 0, 90)]
    assert all(len(t.detect(f)) == 0 for f in frames)


def test_tracker_capture_off_by_default_and_reset():
    img = _small([(2, 3), (2, 5)])
    t = nf.FractalTracker(nf.FractalDetector(CONFIG))
    assert all(len(t.detect(img)) == 0 for _ in range(4))
    t = nf.FractalTracker(nf.FractalDetector(CONFIG), capture_frames=3)
    t.detect(img)
    t.detect(img)
    t.reset()
    assert len(t.detect(img)) == 0  # the streak starts again


def test_capture_frames_checked():
    det = nf.FractalDetector(CONFIG)
    for bad in (-1, 1):
        with pytest.raises(ValueError):
            nf.FractalTracker(det, capture_frames=bad)


def test_capture_on_clutter_finds_nothing():
    from nanofractal.bench import generate_scene

    t = nf.FractalTracker(nf.FractalDetector(CONFIG), capture_frames=2)
    for seed in range(6):
        f = generate_scene(640, 480, "clutter", "FRACTAL_5L_6", seed=seed)
        assert all(len(t.detect(f)) == 0 for _ in range(3))
