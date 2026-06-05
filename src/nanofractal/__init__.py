from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from . import _nanofractal as _nf

__version__ = _nf.__version__
# Diagnostic hook (which OpenCV the extension is linked against). Kept as a
# private module attribute — intentionally NOT part of the public __all__.
_opencv_version = _nf._opencv_version


class Dict(IntEnum):
    ARUCO_MIP_36h12 = 0
    APRILTAG_36h11 = 1


@dataclass
class DetectionResult:
    ids: np.ndarray            # int32 (N,)
    corners: np.ndarray        # float32 (N, 4, 2)
    # points_2d / points_3d are populated only by FractalDetector.detect(
    # ..., with_inner_points=True); they stay None for ArUco and plain fractal
    # detection.
    points_2d: np.ndarray | None = None  # float32 (M, 2)
    points_3d: np.ndarray | None = None  # float32 (M, 3)


class ArucoDetector:
    def __init__(self, dictionary: Dict = Dict.ARUCO_MIP_36h12,
                 max_attempts: int = 1) -> None:
        self.dictionary = Dict(dictionary)
        self._d = _nf.ArucoDetector(int(dictionary), int(max_attempts))

    @property
    def max_attempts(self) -> int:
        return self._d.max_attempts

    def detect(self, image: np.ndarray) -> DetectionResult:
        ids, corners = self._d.detect(image)
        return DetectionResult(ids=ids, corners=corners)

    def detect_batch(self, images, num_threads: int = 0) -> list[DetectionResult]:
        """Detect markers in many images in parallel (0 = use all cores)."""
        results = self._d.detect_batch(list(images), int(num_threads))
        return [DetectionResult(ids=i, corners=c) for i, c in results]

    def estimate_pose(self, corners: np.ndarray, camera_matrix: np.ndarray,
                      dist_coeffs: np.ndarray,
                      marker_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-marker pose (rvecs, tvecs) of shape (N,3) float64 via solvePnP IPPE.

        corners: (N,4,2) as returned by detect().
        """
        corners = np.ascontiguousarray(corners, dtype=np.float32)
        camera_matrix = np.ascontiguousarray(camera_matrix, dtype=np.float64)
        dist_coeffs = np.ascontiguousarray(dist_coeffs, dtype=np.float64)
        return self._d.estimate_pose(corners, camera_matrix, dist_coeffs,
                                     float(marker_size))


_FRACTAL_CONFIGS = {"FRACTAL_2L_6", "FRACTAL_3L_6", "FRACTAL_4L_6", "FRACTAL_5L_6"}


class FractalDetector:
    def __init__(self, config: str, marker_size: float = -1.0) -> None:
        if config not in _FRACTAL_CONFIGS:
            raise ValueError(
                f"invalid config {config!r}; use one of {sorted(_FRACTAL_CONFIGS)}")
        self.config = config
        self.marker_size = float(marker_size)
        self._d = _nf.FractalDetector(config, float(marker_size))

    def detect(self, image: np.ndarray,
               with_inner_points: bool = False) -> DetectionResult:
        if with_inner_points:
            ids, corners, p2d, p3d = self._d.detect_full(image)
            return DetectionResult(ids=ids, corners=corners,
                                   points_2d=p2d, points_3d=p3d)
        ids, corners = self._d.detect(image)
        return DetectionResult(ids=ids, corners=corners)

    def detect_batch(self, images, num_threads: int = 0) -> list[DetectionResult]:
        """Detect fractal markers in many images in parallel (0 = all cores)."""
        results = self._d.detect_batch(list(images), int(num_threads))
        return [DetectionResult(ids=i, corners=c) for i, c in results]

    def estimate_pose(self, result: DetectionResult, camera_matrix: np.ndarray,
                      dist_coeffs: np.ndarray):
        """Pose of the fractal marker as ``(rvec, tvec, reproj_err)`` or ``None``.

        Uses the inner+outer correspondences (``result.points_2d``/``points_3d``
        from ``detect(..., with_inner_points=True)``) when there are >= 4 points
        -- accurate and occlusion-robust -- otherwise falls back to the 4 outer
        corners with a square of side ``marker_size``. Returns ``None`` when no
        marker is visible. ``rvec``/``tvec`` are float64 ``(3,)``; ``reproj_err``
        is the RMS reprojection error in pixels (use it to gate noisy poses).
        """
        cam = np.ascontiguousarray(camera_matrix, dtype=np.float64)
        dist = np.ascontiguousarray(dist_coeffs, dtype=np.float64)
        p2d = result.points_2d
        p3d = result.points_3d
        p2d = (np.ascontiguousarray(p2d, dtype=np.float32)
               if p2d is not None else np.zeros((0, 2), dtype=np.float32))
        p3d = (np.ascontiguousarray(p3d, dtype=np.float32)
               if p3d is not None else np.zeros((0, 3), dtype=np.float32))
        corners = np.ascontiguousarray(result.corners, dtype=np.float32)
        return self._d.estimate_pose(p2d, p3d, corners, cam, dist)

    def draw(self, image: np.ndarray, result: DetectionResult,
             camera_matrix: np.ndarray | None = None,
             dist_coeffs: np.ndarray | None = None,
             rvec: np.ndarray | None = None, tvec: np.ndarray | None = None,
             axis_length: float | None = None) -> np.ndarray:
        """Draw marker outlines (+ ids) on ``image`` in place; with a pose
        (``camera_matrix``, ``dist_coeffs``, ``rvec``, ``tvec``) also draw the
        frame axes. ``image`` must be a writable contiguous uint8 array, BGR for
        colour. ``axis_length`` defaults to half ``marker_size``. Returns ``image``.
        """
        if not image.flags["WRITEABLE"]:
            raise ValueError("image must be writable")
        corners = np.ascontiguousarray(result.corners, dtype=np.float32)
        ids = np.ascontiguousarray(result.ids, dtype=np.int32)
        axes = all(v is not None
                   for v in (camera_matrix, dist_coeffs, rvec, tvec))
        if axes:
            cam = np.ascontiguousarray(camera_matrix, dtype=np.float64)
            dist = np.ascontiguousarray(dist_coeffs, dtype=np.float64)
            rv = np.ascontiguousarray(np.asarray(rvec, dtype=np.float64).reshape(3))
            tv = np.ascontiguousarray(np.asarray(tvec, dtype=np.float64).reshape(3))
            if axis_length is None:
                axis_length = (self.marker_size if self.marker_size > 0 else 1.0) * 0.5
        else:
            cam = dist = rv = tv = np.zeros(0, dtype=np.float64)
            axis_length = 0.0
        self._d.draw(image, corners, ids, bool(axes), cam, dist, rv, tv,
                     float(axis_length))
        return image


__all__ = ["__version__", "Dict", "DetectionResult", "ArucoDetector",
           "FractalDetector"]
