"""estimate_pose: outlier pass, frontal degeneracy, full=True fields (R14)."""

import numpy as np
import pytest

import nanofractal as nf

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
D = np.zeros(5)


def _rot(rvec):
    r = np.asarray(rvec, float)
    a = np.linalg.norm(r)
    if a < 1e-12:
        return np.eye(3)
    k = r / a
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * kx @ kx


def _project(p3, rvec, tvec):
    pc = p3 @ _rot(rvec).T + tvec
    return (pc[:, :2] / pc[:, 2:]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def _result(rvec, tvec, noise=0.0, outliers=0.0, seed=0):
    rng = np.random.default_rng(seed)
    g = np.linspace(-0.1, 0.1, 10)
    p3 = np.array([[x, y, 0.0] for y in g for x in g])
    p2 = _project(p3, rvec, tvec) + rng.normal(0, noise, (len(p3), 2))
    bad = rng.choice(len(p3), int(outliers * len(p3)), replace=False)
    p2[bad] += rng.choice([-1, 1], (len(bad), 2)) * rng.uniform(10, 30, (len(bad), 2))
    h = 0.1
    c3 = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    return nf.DetectionResult(
        ids=np.array([0], np.int32),
        corners=_project(c3, rvec, tvec).astype(np.float32)[None],
        points_2d=p2.astype(np.float32),
        points_3d=p3.astype(np.float32),
    )


def _rot_err_deg(r1, r2):
    c = (np.trace(_rot(r1).T @ _rot(r2)) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


RV = np.array([np.pi - 0.3, 0.1, 0.05])
TV = np.array([0.05, -0.02, 1.0])


@pytest.mark.parametrize("share", [0.1, 0.3])
def test_outliers_are_dropped(share):
    det = nf.FractalDetector("FRACTAL_3L_6", marker_size=0.2)
    clean = det.estimate_pose(_result(RV, TV, noise=0.2), K, D, full=True)
    dirty = det.estimate_pose(_result(RV, TV, noise=0.2, outliers=share), K, D, full=True)
    assert dirty.n_points == 100 and dirty.n_inliers == 100 - int(share * 100)
    assert dirty.reproj_err < 0.5
    assert np.linalg.norm(dirty.tvec - TV) < 2 * np.linalg.norm(clean.tvec - TV) + 1e-3
    assert _rot_err_deg(dirty.rvec, RV) < 2 * _rot_err_deg(clean.rvec, RV) + 0.1


def test_no_outliers_keeps_every_point():
    det = nf.FractalDetector("FRACTAL_3L_6", marker_size=0.2)
    p = det.estimate_pose(_result(RV, TV, noise=0.2), K, D, full=True)
    assert p.n_inliers == p.n_points == 100


def test_half_outliers_is_flagged_by_the_error():
    # the median is itself an outlier: the pass cannot help, the RMS says so
    det = nf.FractalDetector("FRACTAL_3L_6", marker_size=0.2)
    p = det.estimate_pose(_result(RV, TV, noise=0.2, outliers=0.5), K, D, full=True)
    assert p.reproj_err > 5


@pytest.mark.parametrize("use_inner", [True, False])
def test_exactly_frontal(use_inner):
    # rotation exactly pi about x: OpenCV's IPPE alone returns R = I here
    rv = np.array([np.pi, 0.0, 0.0])
    tv = np.array([0.0, 0.0, 1.0])
    res = _result(rv, tv)
    if not use_inner:
        res = nf.DetectionResult(ids=res.ids, corners=res.corners)
    det = nf.FractalDetector("FRACTAL_3L_6", marker_size=0.2)
    p = det.estimate_pose(res, K, D, full=True)
    assert p.reproj_err < 0.1
    assert _rot_err_deg(p.rvec, rv) < 0.05
    np.testing.assert_allclose(p.tvec, tv, atol=1e-4)


def test_relative_error_is_scale_free():
    det = nf.FractalDetector("FRACTAL_3L_6", marker_size=0.2)
    near = det.estimate_pose(_result(RV, TV, noise=0.5), K, D, full=True)
    far = det.estimate_pose(_result(RV, TV * [1, 1, 3], noise=0.5 / 3), K, D, full=True)
    assert far.reproj_err < near.reproj_err / 2
    assert far.reproj_err_rel == pytest.approx(near.reproj_err_rel, rel=0.25)


def test_full_is_a_named_tuple():
    det = nf.FractalDetector("FRACTAL_3L_6", marker_size=0.2)
    res = _result(RV, TV)
    p = det.estimate_pose(res, K, D, full=True)
    assert isinstance(p, nf.Pose) and p.method == "inner"
    rvec, tvec, err = det.estimate_pose(res, K, D)
    np.testing.assert_array_equal(rvec, p.rvec)
