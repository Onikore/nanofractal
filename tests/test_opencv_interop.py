"""TDD: B3 — to_opencv / from_opencv converters. No cv2 import."""
import numpy as np
import pytest

import nanofractal as nf


def _make_result(n: int = 3) -> nf.DetectionResult:
    rng = np.random.default_rng(42)
    ids = np.arange(n, dtype=np.int32)
    corners = rng.uniform(10.0, 100.0, size=(n, 4, 2)).astype(np.float32)
    return nf.DetectionResult(ids=ids, corners=corners)


# ── to_opencv ─────────────────────────────────────────────────────────────────

def test_to_opencv_types():
    result = _make_result(3)
    corners, ids = nf.to_opencv(result)
    assert isinstance(corners, list)
    assert isinstance(ids, np.ndarray)


def test_to_opencv_corners_shape_dtype():
    result = _make_result(3)
    corners, _ = nf.to_opencv(result)
    assert len(corners) == 3
    for c in corners:
        assert isinstance(c, np.ndarray)
        assert c.shape == (1, 4, 2)
        assert c.dtype == np.float32


def test_to_opencv_ids_shape_dtype():
    result = _make_result(3)
    _, ids = nf.to_opencv(result)
    assert ids.shape == (3, 1)
    assert ids.dtype == np.int32


def test_to_opencv_ids_values():
    result = _make_result(4)
    _, ids = nf.to_opencv(result)
    np.testing.assert_array_equal(ids.ravel(), np.arange(4, dtype=np.int32))


def test_to_opencv_corners_values():
    result = _make_result(2)
    corners, _ = nf.to_opencv(result)
    np.testing.assert_allclose(corners[0][0], result.corners[0])
    np.testing.assert_allclose(corners[1][0], result.corners[1])


def test_to_opencv_empty_result():
    empty = nf.DetectionResult(
        ids=np.zeros(0, dtype=np.int32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
    )
    corners, ids = nf.to_opencv(empty)
    assert corners == []
    assert ids.shape == (0, 1)
    assert ids.dtype == np.int32


# ── from_opencv ───────────────────────────────────────────────────────────────

def test_from_opencv_basic_shapes():
    result = _make_result(3)
    corners, ids = nf.to_opencv(result)
    back = nf.from_opencv(corners, ids)
    assert back.ids.shape == (3,)
    assert back.ids.dtype == np.int32
    assert back.corners.shape == (3, 4, 2)
    assert back.corners.dtype == np.float32


def test_from_opencv_none_ids_empty():
    back = nf.from_opencv([], None)
    assert len(back) == 0
    assert back.ids.shape == (0,)
    assert back.corners.shape == (0, 4, 2)


def test_from_opencv_empty_ids_empty():
    back = nf.from_opencv([], np.zeros((0, 1), dtype=np.int32))
    assert len(back) == 0


def test_from_opencv_flat_ids():
    """Accepts (N,) ids as well as (N,1)."""
    rng = np.random.default_rng(7)
    c = rng.uniform(0, 100, (1, 4, 2)).astype(np.float32)
    back = nf.from_opencv([c], np.array([99], dtype=np.int32))
    assert back.ids[0] == 99


def test_from_opencv_flat_corners():
    """Accepts (4,2) corners as well as (1,4,2)."""
    rng = np.random.default_rng(8)
    c_flat = rng.uniform(0, 100, (4, 2)).astype(np.float32)
    back = nf.from_opencv([c_flat], np.array([[42]], dtype=np.int32))
    assert back.corners.shape == (1, 4, 2)
    assert back.ids[0] == 42


def test_from_opencv_points_2d_3d_none():
    result = _make_result(2)
    corners, ids = nf.to_opencv(result)
    back = nf.from_opencv(corners, ids)
    assert back.points_2d is None
    assert back.points_3d is None


# ── round-trip ────────────────────────────────────────────────────────────────

def test_round_trip_preserves_ids():
    result = _make_result(5)
    corners, ids = nf.to_opencv(result)
    back = nf.from_opencv(corners, ids)
    np.testing.assert_array_equal(back.ids, result.ids)


def test_round_trip_preserves_corners():
    result = _make_result(5)
    corners, ids = nf.to_opencv(result)
    back = nf.from_opencv(corners, ids)
    np.testing.assert_allclose(back.corners, result.corners)


def test_round_trip_empty():
    empty = nf.DetectionResult(
        ids=np.zeros(0, dtype=np.int32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
    )
    corners, ids = nf.to_opencv(empty)
    back = nf.from_opencv(corners, ids)
    assert len(back) == 0


# ── in __all__ ────────────────────────────────────────────────────────────────

def test_to_opencv_in_all():
    assert "to_opencv" in nf.__all__


def test_from_opencv_in_all():
    assert "from_opencv" in nf.__all__
