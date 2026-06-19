"""Tests for DetectionResult dunder interface (TASK 1)."""

import numpy as np

import nanofractal as nf
import nanofractal._nanofractal as _nf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(ids, n=None):
    """Build a DetectionResult with synthetic data.

    ids: list[int] — marker ids
    n: number of markers (defaults to len(ids))
    """
    ids_arr = np.array(ids, dtype=np.int32)
    n = len(ids) if n is None else n
    corners = np.zeros((n, 4, 2), dtype=np.float32)
    return nf.DetectionResult(ids=ids_arr, corners=corners)


def _empty():
    """DetectionResult with zero markers."""
    return nf.DetectionResult(
        ids=np.zeros((0,), dtype=np.int32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


def test_len_two():
    r = _make([7, 42])
    assert len(r) == 2


def test_len_empty():
    r = _empty()
    assert len(r) == 0


def test_len_single():
    r = _make([99])
    assert len(r) == 1


# ---------------------------------------------------------------------------
# __bool__
# ---------------------------------------------------------------------------


def test_bool_nonempty_is_true():
    r = _make([7, 42])
    assert bool(r) is True


def test_bool_empty_is_false():
    r = _empty()
    assert bool(r) is False


def test_bool_empty_in_if():
    r = _empty()
    # Make sure the truthiness works naturally in a conditional
    triggered = False
    if r:
        triggered = True
    assert not triggered


def test_bool_nonempty_in_if():
    r = _make([1])
    triggered = False
    if r:
        triggered = True
    assert triggered


# ---------------------------------------------------------------------------
# __iter__
# ---------------------------------------------------------------------------


def test_iter_yields_two_items():
    r = _make([7, 42])
    items = list(r)
    assert len(items) == 2


def test_iter_ids_are_python_ints():
    r = _make([7, 42])
    for marker_id, corners in r:
        assert type(marker_id) is int, f"expected int, got {type(marker_id)}"


def test_iter_corners_shape():
    r = _make([7, 42])
    for marker_id, corners in r:
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float32


def test_iter_ids_match():
    r = _make([7, 42])
    ids_from_iter = [mid for mid, _ in r]
    assert ids_from_iter == [7, 42]


def test_iter_empty_yields_nothing():
    r = _empty()
    items = list(r)
    assert items == []


def test_iter_corners_are_per_marker():
    """Each marker's corners from __iter__ should equal result.corners[i]."""
    ids = [3, 9, 11]
    ids_arr = np.array(ids, dtype=np.int32)
    corners = np.random.default_rng(0).random((3, 4, 2)).astype(np.float32)
    r = nf.DetectionResult(ids=ids_arr, corners=corners)
    for i, (mid, c) in enumerate(r):
        np.testing.assert_array_equal(c, corners[i])


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_contains_n():
    r = _make([7, 42])
    s = repr(r)
    assert "n=2" in s, f"repr={s!r}"


def test_repr_contains_ids():
    r = _make([7, 42])
    s = repr(r)
    assert "7" in s and "42" in s, f"repr={s!r}"


def test_repr_starts_with_DetectionResult():
    r = _make([7, 42])
    s = repr(r)
    assert s.startswith("DetectionResult"), f"repr={s!r}"


def test_repr_no_corners_dump():
    """repr must NOT contain the word 'corners' or huge arrays."""
    r = _make([7, 42])
    s = repr(r)
    assert "corners" not in s, f"repr should not dump corners: {s!r}"


def test_repr_empty():
    r = _empty()
    s = repr(r)
    assert "n=0" in s, f"repr={s!r}"


def test_repr_truncates_many_ids():
    """With >8 ids the repr should truncate."""
    ids = list(range(20))
    r = _make(ids)
    s = repr(r)
    # Should contain ellipsis or '...'
    assert "..." in s, f"expected truncation in repr: {s!r}"
    # Should still show the count
    assert "n=20" in s, f"repr={s!r}"


# ---------------------------------------------------------------------------
# Integration test — real ArUco detection
# ---------------------------------------------------------------------------


def _render_aruco(
    marker_id: int, dictionary: int = 0, cell: int = 40, margin: int = 60
) -> np.ndarray:
    grid = np.asarray(_nf._aruco_marker_image8(dictionary, marker_id))
    marker = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    h, w = marker.shape
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = marker
    return np.ascontiguousarray(canvas)


def test_integration_real_aruco_len_and_iter():
    """Run a real ArUco detection and verify len/__iter__ work on the result."""
    img = _render_aruco(7, dictionary=int(nf.Dict.ARUCO_MIP_36h12))
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)

    # At least one marker found
    assert len(res) >= 1
    assert bool(res) is True

    # __iter__ round-trip
    collected_ids = []
    for mid, corners in res:
        assert type(mid) is int
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float32
        collected_ids.append(mid)

    assert collected_ids == res.ids.tolist()
    assert 7 in collected_ids
