"""Tests for Tasks A, B, C:
A – Clean OpenCV error surfacing as ValueError.
B – SOLVEPNP_IPPE_SQUARE correctness (reprojection guard).
C – opt-in return_reproj parameter on ArucoDetector.estimate_pose.
"""

import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf

CONFIG = "FRACTAL_5L_6"
MARKER_SIZE = 0.05  # metres, ArUco
FRACTAL_SIZE = 0.85  # metres, Fractal


def _rodrigues(rvec):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _project_pinhole(obj, rvec, tvec, cam):
    """Pinhole projection (zero distortion) — independent of cv2 so the wheel
    test environment (pytest + numpy only, no cv2) can run it."""
    R = _rodrigues(rvec)
    P = (R @ np.asarray(obj, dtype=np.float64).T).T + np.asarray(tvec, float).reshape(3)
    x = P[:, 0] / P[:, 2]
    y = P[:, 1] / P[:, 2]
    fx, fy, cx, cy = cam[0, 0], cam[1, 1], cam[0, 2], cam[1, 2]
    return np.stack([fx * x + cx, fy * y + cy], axis=1)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aruco_frontal(render_aruco):
    """Rendered frontal ArUco marker 0; returns (det, corners, cam, dist)."""
    img = render_aruco(0, dictionary=int(nf.Dict.ARUCO_MIP_36h12))
    h, w = img.shape
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    assert 0 in res.ids.tolist(), "synthetic render must be detectable"
    cam = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return det, res.corners, cam, dist


@pytest.fixture
def fractal_frontal():
    """Rendered frontal Fractal marker; returns (det, result, cam, dist)."""
    grid = np.asarray(_nf._fractal_external_image8(CONFIG))
    up = np.kron(grid, np.ones((40, 40), dtype=np.uint8))
    h, w = up.shape
    canvas = np.full((h + 160, w + 160), 255, dtype=np.uint8)
    canvas[80 : 80 + h, 80 : 80 + w] = up
    img = np.ascontiguousarray(canvas)
    det = nf.FractalDetector(CONFIG, marker_size=FRACTAL_SIZE)
    res = det.detect(img)
    assert res.ids.shape[0] > 0, "synthetic render must be detectable"
    cam = np.array(
        [[600.0, 0, (w + 160) / 2], [0, 600.0, (h + 160) / 2], [0, 0, 1]], dtype=np.float64
    )
    dist = np.zeros(5, dtype=np.float64)
    return det, res, cam, dist


# ---------------------------------------------------------------------------
# TASK A – clean error handling
# ---------------------------------------------------------------------------


def test_aruco_pose_nan_camera_raises_value_error(aruco_frontal):
    """NaN values in camera_matrix must raise ValueError, not crash/opaque error."""
    det, corners, _, dist = aruco_frontal
    cam_nan = np.full((3, 3), float("nan"), dtype=np.float64)
    with pytest.raises(ValueError):
        det.estimate_pose(corners, cam_nan, dist, marker_size=MARKER_SIZE)


def test_fractal_pose_nan_camera_raises_value_error(fractal_frontal):
    """NaN values in camera_matrix must raise ValueError on FractalDetector too."""
    det, res, _, dist = fractal_frontal
    cam_nan = np.full((3, 3), float("nan"), dtype=np.float64)
    with pytest.raises(ValueError):
        det.estimate_pose(res, cam_nan, dist)


# ---------------------------------------------------------------------------
# TASK B – SOLVEPNP_IPPE_SQUARE reprojection guard
# ---------------------------------------------------------------------------


def test_aruco_pose_reprojection_error_small(aruco_frontal):
    """Frontal ArUco marker: reprojection RMS with new solver must be < 2 px."""
    det, corners, cam, dist = aruco_frontal
    rvecs, tvecs = det.estimate_pose(corners, cam, dist, marker_size=MARKER_SIZE)

    s = MARKER_SIZE
    obj = np.array(
        [
            [-s / 2, +s / 2, 0.0],
            [+s / 2, +s / 2, 0.0],
            [+s / 2, -s / 2, 0.0],
            [-s / 2, -s / 2, 0.0],
        ],
        dtype=np.float64,
    )

    # Find marker-0 slot in the results (there may be only one marker detected)
    for idx in range(len(rvecs)):
        rvec = rvecs[idx]
        tvec = tvecs[idx]
        proj = _project_pinhole(obj, rvec, tvec, cam)
        detected = corners[idx]
        rms = float(np.sqrt(np.mean(np.sum((proj - detected) ** 2, axis=1))))
        assert rms < 2.0, f"RMS reprojection error too large: {rms:.3f} px (marker slot {idx})"


# ---------------------------------------------------------------------------
# TASK C – opt-in return_reproj
# ---------------------------------------------------------------------------


def test_aruco_return_reproj_true_gives_three_tuple(aruco_frontal):
    """return_reproj=True returns (rvecs, tvecs, errs) as a 3-tuple."""
    det, corners, cam, dist = aruco_frontal
    result = det.estimate_pose(corners, cam, dist, marker_size=MARKER_SIZE, return_reproj=True)
    assert len(result) == 3


def test_aruco_return_reproj_shapes_and_dtype(aruco_frontal):
    """errs must be float64 shape (N,) with finite values < 2 px."""
    det, corners, cam, dist = aruco_frontal
    rvecs, tvecs, errs = det.estimate_pose(
        corners, cam, dist, marker_size=MARKER_SIZE, return_reproj=True
    )
    n = corners.shape[0]
    assert rvecs.shape == (n, 3)
    assert tvecs.shape == (n, 3)
    assert errs.shape == (n,)
    assert errs.dtype == np.float64
    assert np.all(np.isfinite(errs)), f"reproj errs not finite: {errs}"
    assert np.all(errs < 2.0), f"expected small errs for frontal marker: {errs}"


def test_aruco_return_reproj_false_gives_two_tuple(aruco_frontal):
    """return_reproj=False (default) returns exactly 2 values — back-compat."""
    det, corners, cam, dist = aruco_frontal
    result_default = det.estimate_pose(corners, cam, dist, marker_size=MARKER_SIZE)
    assert len(result_default) == 2

    result_explicit = det.estimate_pose(
        corners, cam, dist, marker_size=MARKER_SIZE, return_reproj=False
    )
    assert len(result_explicit) == 2


def test_aruco_return_reproj_empty_input():
    """return_reproj=True with zero markers returns empty arrays without raising."""
    det = nf.ArucoDetector()
    empty = np.zeros((0, 4, 2), dtype=np.float32)
    cam = np.eye(3, dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    rvecs, tvecs, errs = det.estimate_pose(empty, cam, dist, 0.05, return_reproj=True)
    assert rvecs.shape == (0, 3)
    assert tvecs.shape == (0, 3)
    assert errs.shape == (0,)
