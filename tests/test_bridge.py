import numpy as np
import pytest

import nanofractal._nanofractal as _nf


def test_gray_info():
    img = np.zeros((48, 64), dtype=np.uint8)
    rows, cols, ch = _nf._img_info(img)
    assert (rows, cols, ch) == (48, 64, 1)


def test_bgr_info():
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    rows, cols, ch = _nf._img_info(img)
    assert (rows, cols, ch) == (48, 64, 3)


def test_zero_copy_shares_buffer():
    img = np.zeros((48, 64), dtype=np.uint8)
    ptr_cv = _nf._img_data_ptr(img)
    ptr_np = img.__array_interface__["data"][0]
    assert ptr_cv == ptr_np  # no copy: cv::Mat points at the numpy buffer


def test_mean_matches_numpy():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(48, 64), dtype=np.uint8)
    assert _nf._img_mean(img) == pytest.approx(img.mean(), abs=1e-6)


def test_wrong_dtype_raises():
    img = np.zeros((48, 64), dtype=np.float32)
    with pytest.raises(TypeError):
        _nf._img_info(img)


def test_non_contiguous_raises():
    img = np.zeros((48, 128), dtype=np.uint8)[:, ::2]  # non-contiguous view
    assert not img.flags["C_CONTIGUOUS"]
    with pytest.raises(TypeError):
        _nf._img_info(img)


def test_unsupported_channel_count_raises():
    img = np.zeros((48, 64, 4), dtype=np.uint8)  # BGRA / 4 channels not supported
    with pytest.raises(ValueError):
        _nf._img_info(img)


def test_one_dimensional_raises():
    img = np.zeros((64,), dtype=np.uint8)  # not a 2D/3D image
    with pytest.raises(ValueError):
        _nf._img_info(img)


def test_empty_image_is_contiguous():
    # A zero-size dimension is vacuously C-contiguous and must not be misreported
    # as a contiguity (Type) error.
    img = np.zeros((0, 64), dtype=np.uint8)
    rows, cols, ch = _nf._img_info(img)
    assert (rows, cols, ch) == (0, 64, 1)


def test_roundtrip_owned_array():
    out = _nf._echo_corners()  # returns a (2,4,2) float32 array built in C++
    assert out.dtype == np.float32
    assert out.shape == (2, 4, 2)
    assert out[0, 0, 0] == pytest.approx(1.5)
    assert out[1, 3, 1] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Task 3 — (H, W, 1) single-channel input
# ---------------------------------------------------------------------------


def test_hw1_info():
    """(H,W,1) uint8 must be accepted as single-channel grayscale."""
    img = np.zeros((48, 64, 1), dtype=np.uint8)
    rows, cols, ch = _nf._img_info(img)
    assert (rows, cols, ch) == (48, 64, 1)


def test_hw1_detect_finds_marker():
    """Detection on a (H,W,1) array must work and return the expected marker id."""
    import nanofractal as nf

    def _upscale(grid8, cell):
        return np.kron(grid8, np.ones((cell, cell), dtype=np.uint8))

    grid = np.asarray(_nf._aruco_marker_image8(int(nf.Dict.DICT_4X4_50), 7))
    marker = _upscale(grid, 40)
    h, w = marker.shape
    margin = 60
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = marker
    canvas_2d = np.ascontiguousarray(canvas)

    # Make a (H,W,1) version — same buffer layout, just add a channel axis.
    canvas_hwc = np.ascontiguousarray(canvas_2d[..., None])
    assert canvas_hwc.shape == (canvas_2d.shape[0], canvas_2d.shape[1], 1)
    assert canvas_hwc.flags["C_CONTIGUOUS"]

    det = nf.ArucoDetector(nf.Dict.DICT_4X4_50, max_attempts=10)
    res = det.detect(canvas_hwc)
    assert 7 in res.ids.tolist()


def test_unsupported_channel_count_still_raises():
    """4-channel BGRA must still raise ValueError after the (H,W,1) change."""
    img = np.zeros((48, 64, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        _nf._img_info(img)
