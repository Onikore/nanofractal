"""estimate_pose without inner points uses every detected level at its real size (R2)."""

import numpy as np
import pytest

import nanofractal as nf

CONFIG = "FRACTAL_3L_6"
SIZE_M = 0.7  # external side, metres
F = 800.0


def _frontal(img, px_side):
    """Camera looking straight at the marker centre; true distance."""
    h, w = img.shape
    cam = np.array([[F, 0, (w - 1) / 2], [0, F, (h - 1) / 2], [0, 0, 1]], dtype=np.float64)
    return cam, np.zeros(5), F * SIZE_M / px_side


def _render(side_px, crop):
    m = nf.generate_fractal(CONFIG, size_px=side_px)
    pad = 64
    big = np.full((side_px + 2 * pad,) * 2, 255, np.uint8)
    big[pad : pad + side_px, pad : pad + side_px] = m
    c = big.shape[0] // 2
    h = crop // 2
    return np.ascontiguousarray(big[c - h : c + h, c - h : c + h])


@pytest.mark.parametrize("marker_size", [SIZE_M, 0.0])
def test_only_innermost_level_has_true_scale(marker_size):
    side = 2100  # innermost level: 2100 / 7 = 300 px
    img = _render(side, 560)  # level 2 (900 px) is cut off
    det = nf.FractalDetector(CONFIG, marker_size=marker_size)
    res = det.detect(img)
    inner_id = max(i for i, *_ in nf._nanofractal._fractal_markers(CONFIG))
    assert res.ids.tolist() == [inner_id]
    cam, dist, z_true = _frontal(img, side)
    rvec, tvec, err, method, *_ = det.estimate_pose(res, cam, dist, full=True)
    assert method == "corners"
    if marker_size > 0:
        assert tvec[2] == pytest.approx(z_true, rel=0.01)
    else:  # unit external side
        assert tvec[2] == pytest.approx(z_true / SIZE_M, rel=0.01)
    assert abs(tvec[0]) < 0.01 * tvec[2] and abs(tvec[1]) < 0.01 * tvec[2]
    assert err < 1.0


def test_all_levels_used_without_inner_points():
    side = 700
    img = _render(side, side + 100)
    det = nf.FractalDetector(CONFIG, marker_size=SIZE_M)
    res = det.detect(img)
    assert len(res) >= 2
    cam, dist, z_true = _frontal(img, side)
    rvec, tvec, err, method, *_ = det.estimate_pose(res, cam, dist, full=True)
    assert method == "corners"
    assert tvec[2] == pytest.approx(z_true, rel=0.01)
    assert err < 1.0
    # default return stays a 3-tuple
    assert len(det.estimate_pose(res, cam, dist)) == 3


def test_method_inner():
    side = 700
    img = _render(side, side + 100)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    det = nf.FractalDetector(CONFIG, marker_size=SIZE_M)
    res = det.detect(img, with_inner_points=True)
    assert res.points_2d.shape[0] >= 4
    cam, dist, z_true = _frontal(img, side)
    pose = det.estimate_pose(res, cam, dist, full=True)
    method = pose.method
    assert pose.n_points == res.points_2d.shape[0]
    assert method == "inner"
