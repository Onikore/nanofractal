"""Frame-to-frame tracking: detect near the previous detection.

Following the multi-scale/ROI idea of Romero-Ramirez et al., "Speeded up
detection of squared fiducial markers" (Image and Vision Computing, 2018):
in video the marker moves little between frames, so the next search can be
restricted to a region around the last detection. The paper's other lever,
shrinking the image from the previous marker size, is optional
(``auto_scale``): corners are refined on the full-resolution frame, so down to
a factor of 0.35 they are as accurate as without scaling (measured on 4K
synthetic frames); below that they are not, and the factor stops there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from . import DetectionResult, FractalDetector

# detection_scale steps auto_scale picks from (a few fixed steps, so the scale
# stays put while the marker size drifts). Not below 0.35: at 0.25 the outer
# corners of 4K frames came out up to 5 px off, at 0.35 they matched full
# resolution.
_AUTO_SCALES = (1.0, 0.75, 0.6, 0.5, 0.42, 0.35)


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

    ``auto_scale=True`` also shrinks the region (``detection_scale``) so that
    the largest level of the previous frame is about ``scale_target_px``
    pixels wide, never below a factor of 0.35. Corners are still refined on
    the full frame and keep their accuracy, but the smallest inner levels can
    drop out of the result (inner points of every level are still found with
    ``with_inner_points=True``). Only used when the detector's own
    ``detection_scale`` is 1.

    ``use_prior=True`` passes the last non-empty result (at most
    ``prior_max_age`` frames old) as ``prior`` to ``detect``: a level that
    the plain reading misses by a few cells where it was a moment ago is
    kept (``bit_errors > 0``). Flight video, ~2 m: 3 more frames of 203 with
    the marker, 7 more with the outer level; no detection on 600 frames of
    marker-like objects.

    ``capture_frames=N`` (0 = off) picks up a marker before it is ever read:
    while there is no track, a quad that comes as close to a level's code as
    ``prior`` allows (outer levels, a few wrong cells) is a candidate, and
    when candidates of the same level sit in the same place for ``N - 1``
    frames in a row, the next frame reads that place with it as ``prior`` —
    so the first detection is the ``N``-th frame. It implies ``use_prior``
    (a marker picked up this way is rarely read without it). Flight videos (3-4 m,
    2.5-3 px per cell, an overexposed print): the marker is read 8-39 s
    earlier with ``N = 3``; no such candidate in 1242 frames of a flight with
    no readable marker.
    """

    def __init__(
        self,
        detector: FractalDetector,
        margin: float = 0.25,
        pad: int = 16,
        auto_scale: bool = False,
        scale_target_px: float = 400.0,
        use_prior: bool = False,
        prior_max_age: int = 5,
        capture_frames: int = 0,
    ) -> None:
        if margin < 0 or pad < 0:
            raise ValueError("margin and pad must be >= 0")
        if capture_frames < 0 or capture_frames == 1:
            raise ValueError("capture_frames must be 0 (off) or >= 2")
        if scale_target_px <= 0:
            raise ValueError("scale_target_px must be > 0")
        self.detector = detector
        self.margin = float(margin)
        self.pad = int(pad)
        self.auto_scale = bool(auto_scale)
        self.scale_target_px = float(scale_target_px)
        self.use_prior = bool(use_prior)
        self.prior_max_age = int(prior_max_age)
        self.capture_frames = int(capture_frames)
        self._cand: DetectionResult | None = None  # last capture candidate
        self._cand_n = 0  # frames in a row it held its place
        self._last: DetectionResult | None = None  # last non-empty result
        self._last_age = 0
        self._prev: DetectionResult | None = None
        self._prev_scale = 1.0
        self.full_frame_detections = 0  # how often the whole frame was searched

    def reset(self) -> None:
        """Forget the previous detection (e.g. after a scene cut)."""
        self._prev = None
        self._prev_scale = 1.0
        self._last, self._last_age = None, 0
        self._cand, self._cand_n = None, 0

    def _scale(self) -> float:
        """detection_scale for the next region (auto_scale), 1 = none."""
        prev = self._prev
        if not self.auto_scale or prev is None or self.detector.params.detection_scale != 1.0:
            return 1.0
        c = np.asarray(prev.corners, dtype=np.float64)
        side = float(np.linalg.norm(c - np.roll(c, 1, axis=1), axis=2).mean(axis=1).max())
        want = self.scale_target_px / side if side > 0 else 1.0
        return next((s for s in _AUTO_SCALES if s <= want), _AUTO_SCALES[-1])

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

    def _follow_candidate(self, res: DetectionResult | None) -> None:
        """Keep the capture candidate that holds its place (see capture_frames)."""
        if res is None or len(res):
            self._cand, self._cand_n = None, 0
            return
        from . import DetectionResult

        ids, corners, _ = self.detector._d.captured()
        keep = None
        if self._cand is not None:
            pc = self._cand.corners[0].astype(np.float64)
            ps = float(np.linalg.norm(pc - np.roll(pc, 1, axis=0), axis=1).mean())
            for k in range(len(ids)):
                c = corners[k].astype(np.float64)
                side = float(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).mean())
                if (
                    ids[k] == self._cand.ids[0]
                    and np.linalg.norm(c.mean(0) - pc.mean(0)) < ps
                    and ps / 1.5 < side < ps * 1.5
                ):
                    keep = k
                    break
        if keep is not None:
            self._cand_n += 1
        elif len(ids):
            keep, self._cand_n = 0, 1
        else:
            self._cand, self._cand_n = None, 0
            return
        self._cand = DetectionResult(ids=ids[keep : keep + 1], corners=corners[keep : keep + 1])

    def detect(
        self,
        image: np.ndarray,
        with_inner_points: bool = False,
        expected_side_px: tuple[float, float] | None = None,
    ) -> DetectionResult:
        """Like ``detector.detect``, near the last detection (see the class).

        ``expected_side_px=(lo, hi)``: the outer level's side in pixels from
        the height, as in :meth:`FractalDetector.detect`.
        """
        det = self.detector
        prev = self._prev
        roi = self._roi(image.shape)
        prior = None
        use_prior = self.use_prior or self.capture_frames > 0
        if use_prior and self._last is not None and self._last_age <= self.prior_max_age:
            prior = self._last
        capture = self.capture_frames > 0 and prior is None and prev is None
        if capture and self._cand is not None and self._cand_n >= self.capture_frames - 1:
            prior, capture = self._cand, False  # read the candidate's place
        res = None
        scale = 1.0
        if roi is not None and prev is not None:
            p = det.params
            scale = self._scale()
            auto = p.adaptive_block_size == -1
            if auto:
                p.adaptive_block_size = _auto_block_size(
                    image.shape[1], scale if scale < 1.0 else p.detection_scale
                )
            if scale < 1.0:
                p.detection_scale = scale
            try:
                res = det.detect(
                    image,
                    with_inner_points=with_inner_points,
                    roi=roi,
                    prior=prior,
                    expected_side_px=expected_side_px,
                )
            finally:
                if auto:
                    p.adaptive_block_size = -1
                if scale < 1.0:
                    p.detection_scale = 1.0
            # fewer levels than last time: search the whole frame, unless a
            # changed scale explains it (small levels drop out when shrunk)
            if len(res) == 0 or (len(res) < len(prev) and scale == self._prev_scale):
                res = None
        if res is None:
            self.full_frame_detections += 1
            scale = 1.0
            if capture:
                det._d.set_capture(True)
            try:
                res = det.detect(
                    image,
                    with_inner_points=with_inner_points,
                    prior=prior,
                    expected_side_px=expected_side_px,
                )
            finally:
                if capture:
                    det._d.set_capture(False)
        self._follow_candidate(res if capture else None)
        self._prev = res if len(res) else None
        if len(res):
            self._last, self._last_age = res, 1
        else:
            self._last_age += 1
        self._prev_scale = scale
        return res
