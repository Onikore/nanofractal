import numpy as np
import pytest

import nanofractal as nf


def test_blank_image_returns_empty():
    det = nf.ArucoDetector(dictionary=nf.Dict.ARUCO_MIP_36h12, max_attempts=1)
    res = det.detect(np.full((480, 640), 255, dtype=np.uint8))
    assert res.ids.dtype == np.int32
    assert res.ids.shape == (0,)
    assert res.corners.dtype == np.float32
    assert res.corners.shape == (0, 4, 2)


def test_default_max_attempts_is_one():
    det = nf.ArucoDetector()  # realtime default
    assert det.max_attempts == 1


def test_dictionary_is_exposed():
    det = nf.ArucoDetector(dictionary=nf.Dict.APRILTAG_36h11)
    assert det.dictionary == nf.Dict.APRILTAG_36h11


def test_invalid_dictionary_raises():
    with pytest.raises(ValueError):
        nf.ArucoDetector(dictionary=99)


def test_detect_wrong_dtype_raises():
    det = nf.ArucoDetector()
    with pytest.raises(TypeError):
        det.detect(np.zeros((480, 640), dtype=np.float32))


def test_detect_empty_frame_raises():
    det = nf.ArucoDetector()
    with pytest.raises(ValueError):
        det.detect(np.zeros((0, 640), dtype=np.uint8))


@pytest.mark.parametrize("dictionary", [nf.Dict.ARUCO_MIP_36h12, nf.Dict.APRILTAG_36h11])
@pytest.mark.parametrize("marker_id", [0, 1, 7, 42])
def test_detects_rendered_marker(render_aruco, marker_id, dictionary):
    img = render_aruco(marker_id, dictionary=int(dictionary))
    det = nf.ArucoDetector(dictionary, max_attempts=10)
    res = det.detect(img)
    ids = res.ids.tolist()
    assert marker_id in ids
    c = res.corners[ids.index(marker_id)]
    assert c.shape == (4, 2)
    assert (c >= 0).all()
    assert (c[:, 0] < img.shape[1]).all() and (c[:, 1] < img.shape[0]).all()


def test_estimate_pose_frontal_marker(render_aruco):
    img = render_aruco(0, dictionary=int(nf.Dict.ARUCO_MIP_36h12))
    h, w = img.shape
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    assert 0 in res.ids.tolist()

    cam = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5,), dtype=np.float64)
    rvecs, tvecs = det.estimate_pose(res.corners, cam, dist, marker_size=0.05)

    assert rvecs.shape == (len(res.ids), 3)
    assert tvecs.shape == (len(res.ids), 3)
    i = res.ids.tolist().index(0)
    tz = tvecs[i, 2]
    # A centered, frontal marker with the principal point at the image centre
    # projects to the image centre, so its translation is ~on the optical axis.
    assert tz > 0  # in front of the camera
    assert abs(tvecs[i, 0]) < 0.1 * tz  # centered horizontally
    assert abs(tvecs[i, 1]) < 0.1 * tz  # centered vertically
    assert np.isfinite(rvecs[i]).all()


def test_estimate_pose_empty_input():
    det = nf.ArucoDetector()
    empty = np.zeros((0, 4, 2), dtype=np.float32)
    cam = np.eye(3, dtype=np.float64)
    dist = np.zeros((5,), dtype=np.float64)
    rvecs, tvecs = det.estimate_pose(empty, cam, dist, marker_size=0.05)
    assert rvecs.shape == (0, 3)
    assert tvecs.shape == (0, 3)
    assert rvecs.dtype == np.float64 and tvecs.dtype == np.float64
