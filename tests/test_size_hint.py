"""Size hint: detect(..., expected_side_px=(lo, hi)) and lookalikes."""

import numpy as np
import pytest

import nanofractal as nf

CONFIG = "FRACTAL_3L_6"
CELL = 50


def _small(flips=(), px=70, at=(100, 60), shape=(240, 320)):
    """The marker `px` wide (14 cells; inner levels too small to find), outer-grid
    cells (row, col) inverted so the plain reading fails."""
    m = nf.generate_fractal(CONFIG, 14 * CELL).astype(np.float64)
    for r, c in flips:
        blk = m[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL]
        blk[...] = 255 - blk
    k = 14 * CELL // px
    m = m.reshape(px, k, px, k).mean((1, 3))
    page = np.full(shape, 255.0)
    x, y = at
    page[y : y + px, x : x + px] = m
    return page.astype(np.uint8)


def _flips(n):
    cells = [(r, c) for r in (2, 3, 10, 11) for c in range(2, 12)]
    return cells[:n]


DET = nf.FractalDetector(CONFIG)


@pytest.mark.parametrize("n", [10, 14])
def test_hint_reads_more_wrong_cells(n):
    img = _small(_flips(n))
    assert len(DET.detect(img)) == 0
    assert len(DET.detect(img, prior=DET.detect(_small()))) == 0  # beyond the prior tolerance
    r = DET.detect(img, expected_side_px=(55, 90))
    assert r.ids.tolist() == [0] and r.bit_errors.tolist() == [n]
    np.testing.assert_allclose(r.corners[0].mean(0), [135, 95], atol=1.5)
    assert r.lookalikes == []


def test_hint_needs_the_right_size():
    img = _small(_flips(10))
    assert len(DET.detect(img, expected_side_px=(120, 200))) == 0
    small = DET.detect(img, expected_side_px=(20, 45))
    # the outer level is not read with errors; a lower minimum contour lets
    # the (clean) inner level 1 through
    assert 0 not in small.ids and small.bit_errors.tolist() == [0] * len(small)


def test_lookalike_between_the_tolerances():
    r = DET.detect(_small(_flips(18)), expected_side_px=(55, 90))
    assert len(r) == 0
    assert len(r.lookalikes) >= 1
    q = r.lookalikes[0]
    assert isinstance(q, nf.Lookalike)
    assert q.level == 0 and q.bit_errors == 18 and q.border_errors <= 1
    np.testing.assert_allclose(q.centre, [135, 95], atol=2)
    assert q.corners.shape == (4, 2) and q.contrast > 40
    far = DET.detect(_small(_flips(30)), expected_side_px=(55, 90))
    assert len(far) == 0 and far.lookalikes == []


def test_no_hint_no_lookalikes_and_clean_frames_unchanged():
    clean = _small()
    plain = DET.detect(clean)
    assert plain.lookalikes is None
    hinted = DET.detect(clean, expected_side_px=(55, 90))
    np.testing.assert_array_equal(hinted.ids, plain.ids)
    np.testing.assert_array_equal(hinted.corners, plain.corners)
    assert len(DET.detect(_small(_flips(10)))) == 0  # the hint does not leak


def test_hint_with_roi():
    img = _small(_flips(18))
    r = DET.detect(img, roi=(50, 20, 220, 200), expected_side_px=(55, 90))
    np.testing.assert_allclose(r.lookalikes[0].centre, [135, 95], atol=2)
    r = DET.detect(_small(_flips(10)), roi=(50, 20, 220, 200), expected_side_px=(55, 90))
    np.testing.assert_allclose(r.corners[0].mean(0), [135, 95], atol=1.5)


def test_tracker_passes_the_hint():
    img = _small(_flips(10))
    t = nf.FractalTracker(nf.FractalDetector(CONFIG))
    assert len(t.detect(img)) == 0
    assert t.detect(img, expected_side_px=(55, 90)).ids.tolist() == [0]


@pytest.mark.parametrize("bad", [(0, 10), (30, 20), (float("nan"), 5)])
def test_hint_checked(bad):
    with pytest.raises(ValueError):
        DET.detect(_small(), expected_side_px=bad)
    with pytest.raises(TypeError):
        DET.detect(_small(), expected_side_px=5)


def test_hint_on_clutter_finds_nothing():
    from nanofractal.bench import generate_scene

    for seed in range(4):
        f = generate_scene(640, 480, "clutter", "FRACTAL_5L_6", seed=seed)
        for lo in (20, 30, 45, 70):
            r = DET.detect(f, expected_side_px=(lo, lo * 1.7))
            assert len(r) == 0 and r.lookalikes == []
