"""Tests for _validate_camera helper and its integration into estimate_pose (TASK 2)."""

import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf

# ---------------------------------------------------------------------------
# Direct tests on _validate_camera
# ---------------------------------------------------------------------------


def _good_K():
    return np.eye(3, dtype=np.float64)


def _good_D(n=5):
    return np.zeros(n, dtype=np.float64)


def test_validate_camera_valid_passes():
    """A valid (3,3) K + 5-element D must not raise."""
    nf._validate_camera(_good_K(), _good_D(5))


@pytest.mark.parametrize("d_len", [4, 5, 8, 12, 14])
def test_validate_camera_d_allowed_lengths(d_len):
    """All OpenCV-supported distortion-coefficient counts must pass."""
    nf._validate_camera(_good_K(), _good_D(d_len))


def test_validate_camera_K_wrong_shape_2x2():
    with pytest.raises(ValueError, match="3"):
        nf._validate_camera(np.eye(2, dtype=np.float64), _good_D())


def test_validate_camera_K_wrong_shape_3x4():
    with pytest.raises(ValueError, match="3"):
        nf._validate_camera(np.zeros((3, 4), dtype=np.float64), _good_D())


def test_validate_camera_K_1d_raises():
    with pytest.raises(ValueError):
        nf._validate_camera(np.zeros(9, dtype=np.float64), _good_D())


@pytest.mark.parametrize("bad_len", [3, 6, 7, 9, 11, 13])
def test_validate_camera_D_bad_length(bad_len):
    """Unsupported D lengths must raise ValueError mentioning the bad length."""
    with pytest.raises(ValueError, match=str(bad_len)):
        nf._validate_camera(_good_K(), np.zeros(bad_len, dtype=np.float64))


def test_validate_camera_D_2d_raises():
    """D must be 1-D."""
    with pytest.raises(ValueError):
        nf._validate_camera(_good_K(), np.zeros((2, 5), dtype=np.float64))


def test_validate_camera_K_nan_raises():
    K = _good_K()
    K[1, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        nf._validate_camera(K, _good_D())


def test_validate_camera_K_inf_raises():
    K = _good_K()
    K[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        nf._validate_camera(K, _good_D())


def test_validate_camera_D_nan_raises():
    D = _good_D()
    D[2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        nf._validate_camera(_good_K(), D)


def test_validate_camera_D_inf_raises():
    D = _good_D()
    D[0] = float("-inf")
    with pytest.raises(ValueError, match="finite"):
        nf._validate_camera(_good_K(), D)


def test_validate_camera_allowed_lengths_listed_in_message():
    """Error message for bad D length should list the allowed sizes."""
    with pytest.raises(ValueError) as exc_info:
        nf._validate_camera(_good_K(), np.zeros(3, dtype=np.float64))
    msg = str(exc_info.value)
    # At minimum it should mention some valid counts
    assert "4" in msg or "5" in msg or "14" in msg, f"message: {msg!r}"


# ---------------------------------------------------------------------------
# ArucoDetector.estimate_pose integration
# ---------------------------------------------------------------------------


def _aruco_det():
    return nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)


def test_aruco_estimate_pose_empty_corners_valid_cam_passes():
    """Empty corners with a valid camera must not raise (validation runs before solvePnP)."""
    det = _aruco_det()
    empty = np.zeros((0, 4, 2), dtype=np.float32)
    rvecs, tvecs = det.estimate_pose(empty, _good_K(), _good_D(5), marker_size=0.05)
    assert rvecs.shape == (0, 3)
    assert tvecs.shape == (0, 3)


def test_aruco_estimate_pose_K_wrong_shape_raises():
    det = _aruco_det()
    empty = np.zeros((0, 4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        det.estimate_pose(empty, np.eye(2, dtype=np.float64), _good_D(), marker_size=0.05)


def test_aruco_estimate_pose_D_bad_length_raises():
    det = _aruco_det()
    empty = np.zeros((0, 4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        det.estimate_pose(empty, _good_K(), np.zeros(3, dtype=np.float64), marker_size=0.05)


def test_aruco_estimate_pose_K_nan_raises():
    det = _aruco_det()
    empty = np.zeros((0, 4, 2), dtype=np.float32)
    K = _good_K()
    K[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        det.estimate_pose(empty, K, _good_D(), marker_size=0.05)


def test_aruco_estimate_pose_D_nan_raises():
    det = _aruco_det()
    empty = np.zeros((0, 4, 2), dtype=np.float32)
    D = _good_D()
    D[1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        det.estimate_pose(empty, _good_K(), D, marker_size=0.05)


def test_aruco_estimate_pose_D_length_5_passes():
    """Existing tests use 5-element D — must remain valid."""
    det = _aruco_det()
    empty = np.zeros((0, 4, 2), dtype=np.float32)
    rvecs, tvecs = det.estimate_pose(
        empty, _good_K(), np.zeros(5, dtype=np.float64), marker_size=0.05
    )
    assert rvecs.shape == (0, 3)


# ---------------------------------------------------------------------------
# FractalDetector.estimate_pose integration
# ---------------------------------------------------------------------------

CONFIG = "FRACTAL_5L_6"


def _fractal_det():
    return nf.FractalDetector(CONFIG, marker_size=0.85)


def _empty_fractal_result():
    """A DetectionResult with no markers (blank image detection)."""
    det = _fractal_det()
    return det.detect(np.full((480, 640), 255, dtype=np.uint8))


def test_fractal_estimate_pose_no_marker_valid_cam_returns_none():
    """No marker + valid cam must return None (existing behaviour preserved)."""
    det = _fractal_det()
    res = _empty_fractal_result()
    result = det.estimate_pose(res, _good_K(), _good_D(5))
    assert result is None


def test_fractal_estimate_pose_K_wrong_shape_raises():
    det = _fractal_det()
    res = _empty_fractal_result()
    with pytest.raises(ValueError):
        det.estimate_pose(res, np.eye(2, dtype=np.float64), _good_D())


def test_fractal_estimate_pose_D_bad_length_raises():
    det = _fractal_det()
    res = _empty_fractal_result()
    with pytest.raises(ValueError):
        det.estimate_pose(res, _good_K(), np.zeros(6, dtype=np.float64))


def test_fractal_estimate_pose_K_nan_raises():
    det = _fractal_det()
    res = _empty_fractal_result()
    K = _good_K()
    K[2, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        det.estimate_pose(res, K, _good_D())


def test_fractal_estimate_pose_D_inf_raises():
    det = _fractal_det()
    res = _empty_fractal_result()
    D = _good_D()
    D[0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        det.estimate_pose(res, _good_K(), D)


def _render_fractal():
    grid = np.asarray(_nf._fractal_external_image8(CONFIG))
    up = np.kron(grid, np.ones((40, 40), dtype=np.uint8))
    h, w = up.shape
    base = np.full((h + 160, w + 160), 255, dtype=np.uint8)
    base[80 : 80 + h, 80 : 80 + w] = up
    return np.ascontiguousarray(base)


def test_fractal_estimate_pose_real_detection_bad_K_raises():
    """Bad K must raise even when a real marker is present."""
    img = _render_fractal()
    det = _fractal_det()
    res = det.detect(img)
    assert len(res) >= 1  # marker was found
    with pytest.raises(ValueError):
        det.estimate_pose(res, np.eye(2, dtype=np.float64), _good_D())
