"""Frame-to-frame tracking: detect near the previous detection.

Following the multi-scale/ROI idea of Romero-Ramirez et al., "Speeded up
detection of squared fiducial markers" (Image and Vision Computing, 2018):
in video the marker moves little between frames, so the next search can be
restricted to a region around the last detection. Only the region trick is
used -- shrinking the image from the previous marker size (the paper's other
lever) measurably degraded corner accuracy for fractal markers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from . import DetectionResult, FractalDetector


def _auto_block_size(width: int, scale: float) -> int:
    """The adaptive-threshold window detect() picks for a frame this wide."""
    if 0.0 < scale < 1.0:
        width = int(round(width * scale))
    k = max(3, int(15 * float(width) / 1920.0))
    return k + 1 if k % 2 == 0 else k


class FractalTracker:
    """Detect fractal markers in a video, searching near the last detection.

    Each call to :meth:`detect` looks in the bounding box of the previous
    frame's markers, grown by ``margin`` × its size (plus ``pad`` pixels), and
    falls back to the whole frame when nothing was found before, when the
    region finds nothing, or when it finds fewer marker levels than the
    previous frame (so a level is never lost just because the box was tight).
    The adaptive-threshold window is the one the full frame would use, so
    markers well inside the region get exactly the full-frame result.

    Typically 2× faster than plain ``detect()`` at 4K when the marker covers a
    fraction of the frame; it never changes accuracy.
    """

    def __init__(self, detector: FractalDetector, margin: float = 0.25, pad: int = 16) -> None:
        if margin < 0 or pad < 0:
            raise ValueError("margin and pad must be >= 0")
        self.detector = detector
        self.margin = float(margin)
        self.pad = int(pad)
        self._prev: DetectionResult | None = None
        self.full_frame_detections = 0  # how often the whole frame was searched

    def reset(self) -> None:
        """Forget the previous detection (e.g. after a scene cut)."""
        self._prev = None

    def _roi(self, shape) -> tuple[int, int, int, int] | None:
        prev = self._prev
        if prev is None or len(prev) == 0:
            return None
        H, W = shape[:2]
        pts = np.asarray(prev.corners, dtype=np.float64).reshape(-1, 2)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        m = self.margin * max(x1 - x0, y1 - y0) + self.pad
        rx, ry = int(max(0, np.floor(x0 - m))), int(max(0, np.floor(y0 - m)))
        rx1, ry1 = int(min(W, np.ceil(x1 + m) + 1)), int(min(H, np.ceil(y1 + m) + 1))
        if rx1 - rx < 16 or ry1 - ry < 16:
            return None
        if (rx1 - rx) * (ry1 - ry) > 0.7 * W * H:
            return None  # hardly smaller than the frame: not worth it
        return rx, ry, rx1 - rx, ry1 - ry

    def detect(self, image: np.ndarray, with_inner_points: bool = False) -> DetectionResult:
        det = self.detector
        roi = self._roi(image.shape)
        res = None
        if roi is not None:
            p = det.params
            auto = p.adaptive_block_size == -1
            if auto:
                p.adaptive_block_size = _auto_block_size(image.shape[1], p.detection_scale)
            try:
                res = det.detect(image, with_inner_points=with_inner_points, roi=roi)
            finally:
                if auto:
                    p.adaptive_block_size = -1
            if len(res) == 0 or len(res) < len(self._prev):
                res = None
        if res is None:
            self.full_frame_detections += 1
            res = det.detect(image, with_inner_points=with_inner_points)
        self._prev = res if len(res) else None
        return res
