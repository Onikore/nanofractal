"""Pose covariance and the second planar solution (R20)."""

import numpy as np

import nanofractal as nf

CONFIG = "FRACTAL_5L_6"
K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
D = np.zeros(5)


def _rot(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def _project(obj, rv, tv):
    pc = obj @ _rot(rv).T + tv
    uv = pc[:, :2] / pc[:, 2:]
    return uv * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def _level_corners(det):
    """Marker-plane corners (metres) of the detected levels of a clean page."""
    page = np.full((900, 900), 255, np.uint8)
    page[100:800, 100:800] = nf.generate_fractal(CONFIG, 700)
    res = det.detect(page)
    # plane coords from the frontal page: pixel -> metres (0.2 m = 700 px)
    c = (res.corners.reshape(-1, 2) - 450.0 + 0.5) * (0.2 / 700)
    return res.ids, np.c_[c[:, 0], -c[:, 1], np.zeros(len(c))]


def _monte_carlo(det, ids, obj, rv, tv, n=150, sigma=0.5):
    proj = _project(obj, rv, tv)
    rng = np.random.default_rng(0)
    est, covs, poses = [], [], []
    for _ in range(n):
        noisy = (proj + rng.normal(0, sigma, proj.shape)).astype(np.float32)
        p = det.estimate_pose(nf.DetectionResult(ids=ids, corners=noisy.reshape(-1, 4, 2)), K, D, full=True)
        est.append(np.r_[p.rvec, p.tvec])
        covs.append(p.cov)
        poses.append(p)
    return np.array(est), np.array(covs), poses


def test_covariance_matches_scatter():
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    ids, obj = _level_corners(det)
    rv, tv = np.array([0.4, -0.3, 0.1]), np.array([0.05, -0.02, 0.9])
    est, covs, _ = _monte_carlo(det, ids, obj, rv, tv)
    pred = covs.mean(0)
    assert pred.shape == (6, 6)
    np.testing.assert_allclose(pred, pred.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(pred) > 0)
    ratio = np.sqrt(np.diag(np.cov(est.T)) / np.diag(pred))
    assert np.all((ratio > 0.75) & (ratio < 1.35)), ratio


def test_four_corners_have_covariance():
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    ids, obj = _level_corners(det)
    est, covs, _ = _monte_carlo(det, ids[:1], obj[:4], np.array([0.3, 0.2, 0.0]),
                                np.array([0.0, 0.0, 0.8]), n=150)
    ratio = np.sqrt(np.diag(np.cov(est.T)) / np.diag(covs.mean(0)))
    assert np.all((ratio > 0.7) & (ratio < 1.4)), ratio


def test_second_solution_flags_ambiguity():
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    ids, obj = _level_corners(det)
    # strongly tilted, all levels: the mirror solution is far worse
    _, _, poses = _monte_carlo(det, ids, obj, np.array([0.6, -0.3, 0.1]),
                               np.array([0.05, -0.02, 0.9]), n=5)
    for p in poses:
        assert p.rvec_alt is not None and p.reproj_err_alt > 5 * p.reproj_err
    # small, nearly frontal outer level far away: both solutions fit the noise
    _, _, poses = _monte_carlo(det, ids[:1], obj[:4], np.array([0.05, 0.02, 0.0]),
                               np.array([0.0, 0.0, 6.0]), n=20)
    ratios = [p.reproj_err_alt / p.reproj_err for p in poses if p.rvec_alt is not None]
    assert len(ratios) >= 15 and np.median(ratios) < 2.0
    # the alternative is a genuinely different rotation
    p = poses[0]
    assert np.linalg.norm(p.rvec_alt - p.rvec) > 1e-3


def test_plain_return_unchanged():
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    ids, obj = _level_corners(det)
    proj = _project(obj, np.array([0.3, 0.1, 0.0]), np.array([0.0, 0.0, 0.8])).astype(np.float32)
    out = det.estimate_pose(nf.DetectionResult(ids=ids, corners=proj.reshape(-1, 4, 2)), K, D)
    assert len(out) == 3


def _project_fisheye(obj, rv, tv, k):
    pc = obj @ _rot(rv).T + tv
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    r = np.hypot(x, y)
    th = np.arctan(r)
    t2 = th * th
    thd = th * (1 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4)
    sc = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
    return np.c_[K[0, 0] * x * sc + K[0, 2], K[1, 1] * y * sc + K[1, 2]]


def test_fisheye_covariance_matches_scatter():
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    ids, obj = _level_corners(det)
    k = np.array([0.05, -0.01, 0.0, 0.0])
    proj = _project_fisheye(obj, np.array([0.4, -0.3, 0.1]), np.array([0.05, -0.02, 0.9]), k)
    rng = np.random.default_rng(2)
    est, covs = [], []
    for _ in range(150):
        noisy = (proj + rng.normal(0, 0.5, proj.shape)).astype(np.float32)
        r = nf.DetectionResult(ids=ids, corners=noisy.reshape(-1, 4, 2))
        p = det.estimate_pose(r, K, k, fisheye=True, full=True)
        est.append(np.r_[p.rvec, p.tvec])
        covs.append(p.cov)
    ratio = np.sqrt(np.diag(np.cov(np.array(est).T)) / np.diag(np.mean(covs, axis=0)))
    assert np.all((ratio > 0.75) & (ratio < 1.35)), ratio
