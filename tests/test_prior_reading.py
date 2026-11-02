"""Reading with a prior and per-level reading quality."""

import numpy as np
import pytest

import nanofractal as nf

CONFIG = "FRACTAL_3L_6"
CELL = 50  # px per cell of the outer level (14 cells, 700 px)


def _page(flips=()):
    """The marker at 50 px per cell; `flips` = outer-grid cells (row, col) inverted."""
    m = nf.generate_fractal(CONFIG, 14 * CELL)
    for r, c in flips:
        blk = m[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL]
        blk[...] = 255 - blk
    page = np.full((900, 900), 255, np.uint8)
    page[100:800, 100:800] = m
    return page


DET = nf.FractalDetector(CONFIG)
CLEAN = DET.detect(_page())


def test_quality_fields():
    r = CLEAN
    assert r.ids.tolist() == [0, 1, 2]
    np.testing.assert_allclose(r.cell_px[0], CELL, rtol=0.02)
    assert np.all(r.contrast > 200)
    assert r.bit_errors.tolist() == [0, 0, 0]
    full = DET.detect(_page(), with_inner_points=True)
    np.testing.assert_array_equal(full.bit_errors, r.bit_errors)
    np.testing.assert_allclose(full.cell_px, r.cell_px)
    one = DET.detect(_page(), all_instances=True).instance(0)
    np.testing.assert_allclose(one.cell_px, r.cell_px)


@pytest.mark.parametrize("n", [1, 3, 8])
def test_prior_accepts_a_few_wrong_cells(n):
    img = _page([(2, 2 + k) for k in range(n)])
    assert 0 not in DET.detect(img).ids  # the plain reading needs every cell
    r = DET.detect(img, prior=CLEAN)
    assert 0 in r.ids
    k = r.ids.tolist().index(0)
    assert r.bit_errors[k] == n
    np.testing.assert_allclose(r.corners[k], CLEAN.corners[0], atol=0.5)


def test_prior_rejects_too_many_wrong_cells():
    img = _page([(2, 2 + k) for k in range(10)])
    assert 0 not in DET.detect(img, prior=CLEAN).ids


def test_prior_elsewhere_is_ignored():
    img = _page([(2, 3), (2, 5)])
    moved = nf.DetectionResult(ids=CLEAN.ids, corners=CLEAN.corners + 600)
    assert 0 not in DET.detect(img, prior=moved).ids
    small = nf.DetectionResult(ids=CLEAN.ids, corners=(CLEAN.corners - 450) * 0.5 + 450)
    assert 0 not in DET.detect(img, prior=small).ids


def test_prior_does_not_leak_and_keeps_clean_frames():
    img = _page([(2, 3), (2, 5)])
    assert 0 in DET.detect(img, prior=CLEAN).ids
    assert 0 not in DET.detect(img).ids  # cleared after the call
    again = DET.detect(_page(), prior=CLEAN)
    np.testing.assert_array_equal(again.corners, CLEAN.corners)
    np.testing.assert_array_equal(again.bit_errors, [0, 0, 0])


def test_prior_with_roi():
    img = _page([(2, 3), (2, 5)])
    r = DET.detect(img, roi=(50, 50, 800, 800), prior=CLEAN)
    assert 0 in r.ids
    k = r.ids.tolist().index(0)
    np.testing.assert_allclose(r.corners[k], CLEAN.corners[0], atol=0.5)


def test_prior_on_clutter_finds_nothing():
    rng = np.random.default_rng(1)
    for _ in range(5):
        img = rng.integers(0, 256, (720, 1280), dtype=np.uint8)
        img = np.ascontiguousarray(np.repeat(np.repeat(img[::8, ::8], 8, 0), 8, 1))
        quads = DET.diagnose(img).quads
        if not quads:
            continue
        cs = np.array([q.corners for q in quads], np.float32)
        prior = nf.DetectionResult(ids=np.zeros(len(cs), np.int32), corners=cs.reshape(-1, 4, 2))
        assert len(DET.detect(img, prior=prior)) == 0


def test_prior_type_checked():
    with pytest.raises(TypeError):
        DET.detect(_page(), prior=(0, 1))


def test_tracker_use_prior():
    img = _page([(2, 3), (2, 5)])
    plain = nf.FractalTracker(DET)
    plain.detect(_page())
    assert 0 not in plain.detect(img).ids
    tr = nf.FractalTracker(DET, use_prior=True)
    tr.detect(_page())
    r = tr.detect(img)
    assert 0 in r.ids and r.bit_errors[r.ids.tolist().index(0)] == 2
    # too old: no prior after prior_max_age empty frames
    tr = nf.FractalTracker(DET, use_prior=True, prior_max_age=1)
    tr.detect(_page())
    blank = np.full_like(img, 255)
    tr.detect(blank)
    tr.detect(blank)
    assert 0 not in tr.detect(img).ids
