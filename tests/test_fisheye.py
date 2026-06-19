import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf

CONFIG = "FRACTAL_5L_6"


# --- a reference OpenCV-fisheye projector (numpy), used to synthesise exact
#     fisheye observations from a known pose so we can verify pose recovery. ---
def _rodrigues(rvec):
    rvec = np.asarray(rvec, dtype=np.float64)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _fisheye_project(obj, rvec, tvec, K, D):
    """OpenCV fisheye model: theta_d = theta*(1+k1 th^2+k2 th^4+k3 th^6+k4 th^8)."""
    R = _rodrigues(rvec)
    P = (R @ np.asarray(obj, dtype=np.float64).T).T + np.asarray(tvec, float)
    x = P[:, 0] / P[:, 2]
    y = P[:, 1] / P[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    k1, k2, k3, k4 = D
    theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    scale = np.where(r > 1e-12, theta_d / r, 1.0)
    xp, yp = x * scale, y * scale
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = fx * xp + cx, fy * yp + cy
    return np.stack([u, v], axis=1).astype(np.float32)


def _aruco_scene(marker_id=0, cell=40, margin=80):
    grid = np.asarray(_nf._aruco_marker_image8(int(nf.Dict.DICT_4X4_50), marker_id))
    up = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    h, w = up.shape
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = up
    return np.ascontiguousarray(canvas)


def _cam(img, f=600.0):
    h, w = img.shape[:2]
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_fisheye_requires_4_coeffs_aruco():
    img = _aruco_scene(7)
    det = nf.ArucoDetector(nf.Dict.DICT_4X4_50, max_attempts=10)
    res = det.detect(img)
    with pytest.raises(ValueError, match="fisheye"):
        det.estimate_pose(res.corners, _cam(img), np.zeros(5), 0.05, fisheye=True)


def test_fisheye_requires_4_coeffs_fractal():
    img = _aruco_scene(7)  # any detection result; the check is on dist length
    fdet = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = nf.DetectionResult(ids=np.zeros(0, np.int32), corners=np.zeros((0, 4, 2), np.float32))
    with pytest.raises(ValueError, match="fisheye"):
        fdet.estimate_pose(res, _cam(img), np.zeros(5), fisheye=True)


# ---------------------------------------------------------------------------
# ground-truth round-trip: synthesise exact fisheye corners, recover the pose
# ---------------------------------------------------------------------------


def test_fisheye_roundtrip_recovers_pose_aruco():
    s = 0.1
    obj = np.array(
        [[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]],
        dtype=np.float64,
    )
    rvec_gt = np.array([0.05, -0.08, 0.02])
    tvec_gt = np.array([0.03, -0.02, 0.6])
    K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]], dtype=np.float64)
    D = np.array([-0.10, 0.02, 0.0, 0.0], dtype=np.float64)  # fisheye k1..k4

    corners = _fisheye_project(obj, rvec_gt, tvec_gt, K, D).reshape(1, 4, 2)
    det = nf.ArucoDetector(nf.Dict.DICT_4X4_50)
    rv, tv, errs = det.estimate_pose(corners, K, D, s, return_reproj=True, fisheye=True)
    np.testing.assert_allclose(tv[0], tvec_gt, atol=3e-3)
    np.testing.assert_allclose(rv[0], rvec_gt, atol=2e-2)
    assert errs[0] < 0.5  # data is exact → near-zero reprojection error


def test_fisheye_roundtrip_recovers_pose_fractal():
    # Use real inner-point object coords from a detection, re-projected through
    # the fisheye model at a known pose, then recover that pose.
    grid = np.asarray(_nf._fractal_external_image8(CONFIG))
    up = np.kron(grid, np.ones((40, 40), dtype=np.uint8))
    h, w = up.shape
    base = np.full((h + 160, w + 160), 255, dtype=np.uint8)
    base[80 : 80 + h, 80 : 80 + w] = up
    rng = np.random.default_rng(0)
    img = np.clip(base.astype(np.float32) + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)

    det = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = det.detect(img, with_inner_points=True)
    assert res.points_3d is not None and res.points_3d.shape[0] >= 4
    p3d = np.ascontiguousarray(res.points_3d, dtype=np.float64)

    rvec_gt = np.array([0.04, 0.03, -0.02])
    tvec_gt = np.array([0.05, -0.03, 1.2])
    K = np.array([[700.0, 0, 360], [0, 700.0, 360], [0, 0, 1]], dtype=np.float64)
    D = np.array([-0.08, 0.01, 0.0, 0.0], dtype=np.float64)
    p2d_syn = _fisheye_project(p3d, rvec_gt, tvec_gt, K, D)

    res2 = nf.DetectionResult(
        ids=res.ids, corners=res.corners, points_2d=p2d_syn, points_3d=res.points_3d
    )
    out = det.estimate_pose(res2, K, D, fisheye=True)
    assert out is not None
    rv, tv, err = out
    np.testing.assert_allclose(tv, tvec_gt, atol=5e-3)
    np.testing.assert_allclose(rv, rvec_gt, atol=3e-2)
    assert err < 0.5


def test_fisheye_reproj_finite_on_real_corners_aruco():
    # Real (pinhole-rendered) corners treated as fisheye: the undistort→solve→
    # fisheye-reproject loop is self-consistent, so the error stays finite/small.
    img = _aruco_scene(7)
    det = nf.ArucoDetector(nf.Dict.DICT_4X4_50, max_attempts=10)
    res = det.detect(img)
    rv, tv, errs = det.estimate_pose(
        res.corners, _cam(img), np.zeros(4), 0.05, return_reproj=True, fisheye=True
    )
    assert errs.shape == (len(res),)
    assert np.isfinite(errs).all()
