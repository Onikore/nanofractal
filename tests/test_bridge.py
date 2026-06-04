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
