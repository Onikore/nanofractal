"""Large-image and memory-safety tests for nanofractal.

Dependencies: numpy + nanofractal only — NO cv2.

Frame sizes
-----------
4K  : 3840 × 2160   (~8 MP)
8K  : 7680 × 4320   (~33 MP)   — single detection only, no loop
1080p: 1920 × 1080  — used in the stability / leak-guard loop

The 8K tests allocate large buffers (≈32 MB gray, ≈96 MB BGR) but each
performs exactly one detection to keep the file's total runtime within a
few seconds.  The stability test (50 × detect + detect_batch on 1080p) is
the scenario that runs meaningfully under AddressSanitizer in CI to catch
heap errors and leaks.
"""

from __future__ import annotations

import numpy as np

import nanofractal as nf

# ---------------------------------------------------------------------------
# Shared detector and marker parameters
# ---------------------------------------------------------------------------
_DICT = nf.Dict.ARUCO_MIP_36h12
_MARKER_ID_4K = 5
_MARKER_ID_8K = 3
_MARKER_MARGIN = 80


def _make_detector(attempts: int = 10) -> nf.ArucoDetector:
    return nf.ArucoDetector(_DICT, max_attempts=attempts)


def _render_marker(marker_id: int, size_px: int = 300) -> np.ndarray:
    """Return a square grayscale ArUco marker via nf.generate_aruco (no cv2)."""
    return nf.generate_aruco(marker_id, size_px=size_px, dictionary=_DICT)


def _frame_gray(H: int, W: int, marker: np.ndarray, margin: int = _MARKER_MARGIN) -> np.ndarray:
    """White HxW uint8 grayscale frame with marker pasted near the top-left corner."""
    frame = np.full((H, W), 255, dtype=np.uint8)
    s = marker.shape[0]
    frame[margin : margin + s, margin : margin + s] = marker
    return np.ascontiguousarray(frame)


def _frame_bgr(H: int, W: int, marker: np.ndarray, margin: int = _MARKER_MARGIN) -> np.ndarray:
    """White HxWx3 uint8 BGR frame with marker pasted near the top-left corner."""
    frame = np.full((H, W, 3), 255, dtype=np.uint8)
    s = marker.shape[0]
    frame[margin : margin + s, margin : margin + s, :] = marker[:, :, np.newaxis]
    return np.ascontiguousarray(frame)


# ---------------------------------------------------------------------------
# 4K (3840 × 2160) — grayscale and BGR
# ---------------------------------------------------------------------------


class Test4KGray:
    """Single-shot detection on a 4K (3840×2160) grayscale frame."""

    H, W = 2160, 3840

    def test_detect_returns_without_error(self):
        det = _make_detector(attempts=1)
        blank = np.full((self.H, self.W), 200, dtype=np.uint8)
        res = det.detect(blank)
        assert res.ids.dtype == np.int32
        assert res.corners.dtype == np.float32

    def test_marker_found(self):
        det = _make_detector()
        marker = _render_marker(_MARKER_ID_4K, size_px=300)
        frame = _frame_gray(self.H, self.W, marker)
        res = det.detect(frame)
        assert _MARKER_ID_4K in res.ids.tolist(), (
            f"marker id {_MARKER_ID_4K} not detected in 4K gray; got {res.ids.tolist()}"
        )


class Test4KBGR:
    """Single-shot detection on a 4K (3840×2160) BGR frame."""

    H, W = 2160, 3840

    def test_detect_returns_without_error(self):
        det = _make_detector(attempts=1)
        blank = np.full((self.H, self.W, 3), 200, dtype=np.uint8)
        res = det.detect(blank)
        assert res.ids.dtype == np.int32
        assert res.corners.dtype == np.float32

    def test_marker_found(self):
        det = _make_detector()
        marker = _render_marker(_MARKER_ID_4K, size_px=300)
        frame = _frame_bgr(self.H, self.W, marker)
        res = det.detect(frame)
        assert _MARKER_ID_4K in res.ids.tolist(), (
            f"marker id {_MARKER_ID_4K} not detected in 4K BGR; got {res.ids.tolist()}"
        )


# ---------------------------------------------------------------------------
# 8K (7680 × 4320) — single detection only (large buffers; no loop)
# ---------------------------------------------------------------------------


class Test8KGray:
    """Single-shot detection on an 8K (7680×4320) grayscale frame.

    Each test performs exactly one detect() call — no loops — to keep
    memory pressure and runtime reasonable in both local and CI runs.
    """

    H, W = 4320, 7680

    def test_detect_returns_without_error(self):
        det = _make_detector(attempts=1)
        blank = np.full((self.H, self.W), 200, dtype=np.uint8)
        res = det.detect(blank)
        assert res.ids.dtype == np.int32
        assert res.corners.dtype == np.float32

    def test_marker_found(self):
        det = _make_detector()
        marker = _render_marker(_MARKER_ID_8K, size_px=400)
        frame = _frame_gray(self.H, self.W, marker)
        res = det.detect(frame)
        assert _MARKER_ID_8K in res.ids.tolist(), (
            f"marker id {_MARKER_ID_8K} not detected in 8K gray; got {res.ids.tolist()}"
        )


class Test8KBGR:
    """Single-shot detection on an 8K (7680×4320) BGR frame.

    Each test performs exactly one detect() call — no loops — to keep
    memory pressure and runtime reasonable in both local and CI runs.
    """

    H, W = 4320, 7680

    def test_detect_returns_without_error(self):
        det = _make_detector(attempts=1)
        blank = np.full((self.H, self.W, 3), 200, dtype=np.uint8)
        res = det.detect(blank)
        assert res.ids.dtype == np.int32
        assert res.corners.dtype == np.float32

    def test_marker_found(self):
        det = _make_detector()
        marker = _render_marker(_MARKER_ID_8K, size_px=400)
        frame = _frame_bgr(self.H, self.W, marker)
        res = det.detect(frame)
        assert _MARKER_ID_8K in res.ids.tolist(), (
            f"marker id {_MARKER_ID_8K} not detected in 8K BGR; got {res.ids.tolist()}"
        )


# ---------------------------------------------------------------------------
# Stability / allocator-stress guard (1080p, 50 iterations)
#
# This test is fast (~0.5 s locally) and runs meaningfully under
# AddressSanitizer in CI: repeated alloc/free cycles expose heap corruption
# and use-after-free bugs that a single call would miss.
# ---------------------------------------------------------------------------


def test_repeated_detect_stability():
    """50 × detect() on a 1080p frame — allocator stress / leak guard."""
    H, W = 1080, 1920
    det = _make_detector(attempts=1)
    frame = np.ascontiguousarray(np.full((H, W), 200, dtype=np.uint8))
    for _ in range(50):
        res = det.detect(frame)
        assert res.ids.dtype == np.int32
        assert res.corners.dtype == np.float32


def test_repeated_detect_batch_stability():
    """50 × detect_batch() on a 1080p frame — allocator stress / leak guard."""
    H, W = 1080, 1920
    det = _make_detector(attempts=1)
    frame = np.ascontiguousarray(np.full((H, W), 200, dtype=np.uint8))
    batch = [frame] * 4
    for _ in range(50):
        results = det.detect_batch(batch, num_threads=2)
        assert len(results) == 4
        for res in results:
            assert res.ids.dtype == np.int32
            assert res.corners.dtype == np.float32
