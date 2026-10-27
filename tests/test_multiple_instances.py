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


# --- R8: every copy, grouped into instances --------------------------------


def _copies(sizes, gap=80):
    h = max(sizes) + 120
    page = np.full((h, sum(sizes) + gap * (len(sizes) + 1)), 255, np.uint8)
    x = gap
    for s in sizes:
        page[60 : 60 + s, x : x + s] = nf.generate_fractal(CONFIG, s)
        x += s + gap
    return np.ascontiguousarray(page)


@pytest.mark.parametrize("sizes", [[300, 200], [240, 320, 180]])
def test_all_instances_grouped(sizes):
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    page = _copies(sizes)
    res = det.detect(page, all_instances=True)
    assert res.n_instances == len(sizes) and res.multiple_instances
    order = np.argsort(sizes)[::-1]  # copy 0 = the largest
    gap, x = 80, [80 + sum(sizes[:i]) + 80 * i for i in range(len(sizes))]
    for k in range(res.n_instances):
        one = res.instance(k)
        i = order[k]
        cx = one.corners[0][:, 0].mean()
        assert x[i] < cx < x[i] + sizes[i]  # the right copy
        # the same levels and corners as detecting that copy alone
        ref = det.detect(_copies([sizes[i]]))
        assert one.ids.tolist() == ref.ids.tolist() and len(ref) >= 2
        np.testing.assert_allclose(one.corners - [x[i] - gap, 0], ref.corners, atol=0.05)


def test_all_instances_pose_per_copy():
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    res = det.detect(_copies([300, 200]), with_inner_points=True, all_instances=True)
    assert res.points_instance is not None
    assert res.points_instance.shape == (len(res.points_2d),)
    K = np.array([[900.0, 0, 400], [0, 900.0, 210], [0, 0, 1]])
    poses = [det.estimate_pose(res.instance(k), K, np.zeros(5)) for k in range(2)]
    for p in poses:
        assert p is not None and p[2] < 0.5
    # the big copy is left of the small one and closer to the camera
    assert poses[0][1][0] < poses[1][1][0] and poses[0][1][2] < poses[1][1][2]
    with pytest.raises(ValueError, match="instance"):
        det.estimate_pose(res, K, np.zeros(5))


def test_all_instances_single_marker_unchanged():
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    page = _copies([300])
    a = det.detect(page, with_inner_points=True)
    b = det.detect(page, with_inner_points=True, all_instances=True)
    assert a.instances is None and a.n_instances == 1
    assert b.n_instances == 1 and not b.multiple_instances
    np.testing.assert_array_equal(a.ids, b.ids)
    np.testing.assert_array_equal(a.corners, b.corners)
    np.testing.assert_array_equal(a.points_2d, b.points_2d)
    with pytest.raises(ValueError):
        a.instance(0)
