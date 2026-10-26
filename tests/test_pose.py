"""FractalDetector.estimate_pose: clean error surfacing and reprojection guard."""

import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf

CONFIG = "FRACTAL_5L_6"
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


def test_fractal_pose_nan_camera_raises_value_error(fractal_frontal):
    """NaN values in camera_matrix must raise ValueError on FractalDetector too."""
    det, res, _, dist = fractal_frontal
    cam_nan = np.full((3, 3), float("nan"), dtype=np.float64)
    with pytest.raises(ValueError):
        det.estimate_pose(res, cam_nan, dist)


# ---------------------------------------------------------------------------
# Outer-corner fallback (IPPE_SQUARE) reprojection guard
# ---------------------------------------------------------------------------


def test_fractal_outer_corner_pose_reprojects(fractal_frontal):
    """Frontal marker: independent pinhole reprojection of the 4 outer corners < 2 px."""
    det, res, cam, dist = fractal_frontal
    rvec, tvec, err = det.estimate_pose(res, cam, dist)
    h = FRACTAL_SIZE / 2
    obj = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)
    proj = _project_pinhole(obj, rvec, tvec, cam)
    rms = float(np.sqrt(np.mean(np.sum((proj - res.corners[0]) ** 2, axis=1))))
    assert rms < 2.0, f"RMS reprojection error too large: {rms:.3f} px"
    assert err == pytest.approx(rms, abs=1e-3)
