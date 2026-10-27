"""FractalDetector.diagnose: why each candidate quad was kept or dropped (R16)."""

import numpy as np
import pytest

import nanofractal as nf

CONFIG = "FRACTAL_3L_6"
SIDE = 300


def _page(flip_cell=None):
    page = np.full((SIDE + 120, SIDE + 200), 255, np.uint8)
    page[60 : 60 + SIDE, 100 : 100 + SIDE] = nf.generate_fractal(CONFIG, SIDE)
    if flip_cell is not None:  # invert one cell of the outer (14 x 14) grid
        cell = SIDE / 14
        r, c = flip_cell
        y0, x0 = 60 + round(cell * r), 100 + round(cell * c)
        blk = page[y0 + 3 : y0 + int(cell) - 3, x0 + 3 : x0 + int(cell) - 3]
        blk[:] = 255 - blk
    return page


@pytest.mark.parametrize("scale", [1.0, 0.75, 0.5])
def test_clean_marker(scale):
    p = nf.DetectorParams()
    p.detection_scale = scale
    det = nf.FractalDetector(CONFIG, marker_size=0.2, params=p)
    page = _page()
    d = det.diagnose(page)
    res = det.detect(page)
    np.testing.assert_array_equal(d.result.ids, res.ids)
    np.testing.assert_array_equal(d.result.corners, res.corners)
    acc = [q for q in d.quads if q.reason == "accepted"]
    assert sorted(q.nearest_id for q in acc) == sorted(res.ids.tolist())
    for q in acc:
        assert q.bit_errors == 0 and q.border_errors == 0 and q.contrast > 100
        i = res.ids.tolist().index(q.nearest_id)
        assert np.abs(q.corners - res.corners[i]).max() < 3  # before refinement
    assert d.n_contours == len(d.quads) + d.n_not_quad
    assert "accepted" in repr(d)


@pytest.mark.parametrize("col", [1, 6, 12])
def test_one_wrong_bit_reported(col):
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    d = det.diagnose(_page(flip_cell=(1, col)))
    assert 0 not in d.result.ids.tolist()  # the outer level is lost ...
    outer = [q for q in d.quads if q.cells == 14]
    assert outer
    for q in outer:  # ... and the report says why: one bit off id 0
        assert q.reason == "bits"
        assert q.nearest_id == 0 and q.bit_errors == 1 and q.border_errors == 0


def test_roi_offsets_quads():
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    page = _page()
    full = det.diagnose(page)
    roi = det.diagnose(page, roi=(50, 30, SIDE + 120, SIDE + 60))
    np.testing.assert_array_equal(roi.result.ids, full.result.ids)
    a = sorted((q.nearest_id, q.corners.sum()) for q in full.quads if q.reason == "accepted")
    b = sorted((q.nearest_id, q.corners.sum()) for q in roi.quads if q.reason == "accepted")
    assert [i for i, _ in a] == [i for i, _ in b]
    np.testing.assert_allclose([s for _, s in a], [s for _, s in b], atol=1e-3)


def test_blank_frame():
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    d = det.diagnose(np.full((240, 320), 128, np.uint8))
    assert len(d.result) == 0 and d.quads == []
