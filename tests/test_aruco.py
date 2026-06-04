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


@pytest.mark.parametrize("marker_id", [0, 1, 7, 42])
def test_detects_rendered_marker(render_aruco, marker_id):
    img = render_aruco(marker_id, dictionary=int(nf.Dict.ARUCO_MIP_36h12))
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    assert marker_id in res.ids.tolist()
    idx = res.ids.tolist().index(marker_id)
    c = res.corners[idx]
    assert c.shape == (4, 2)
    assert (c >= 0).all()
    assert (c[:, 0] < img.shape[1]).all() and (c[:, 1] < img.shape[0]).all()
