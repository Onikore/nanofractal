"""DetectionResult.multiple_instances: two copies of one marker in view (R8a)."""

import numpy as np
import pytest

import nanofractal as nf

CONFIG = "FRACTAL_3L_6"


def _page(n_copies, side=300):
    m = nf.generate_fractal(CONFIG, side)
    page = np.full((side + 120, n_copies * (side + 100) + 100), 255, np.uint8)
    for k in range(n_copies):
        x = 100 + k * (side + 100)
        page[60 : 60 + side, x : x + side] = m
    return np.ascontiguousarray(page)


@pytest.mark.parametrize("inner", [False, True])
def test_two_copies_flagged(inner):
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    res = det.detect(_page(2), with_inner_points=inner)
    assert res.multiple_instances
    assert len(set(res.ids.tolist())) == len(res)  # still one entry per level


@pytest.mark.parametrize("inner", [False, True])
def test_one_copy_not_flagged(inner):
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    res = det.detect(_page(1), with_inner_points=inner)
    assert len(res) >= 2
    assert not res.multiple_instances


def test_batch_and_roi():
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    one, two = _page(1), _page(2)
    r1, r2 = det.detect_batch([one, two])
    assert not r1.multiple_instances and r2.multiple_instances
    # an ROI around the first copy sees one marker
    assert not det.detect(two, roi=(0, 0, 450, two.shape[0])).multiple_instances


def test_detection_scale():
    p = nf.DetectorParams()
    p.detection_scale = 0.5
    det = nf.FractalDetector(CONFIG, marker_size=0.2, params=p)
    assert not det.detect(_page(1)).multiple_instances
    assert det.detect(_page(2)).multiple_instances
