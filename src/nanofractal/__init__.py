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

    def estimate_pose(self, corners: np.ndarray, camera_matrix: np.ndarray,
                      dist_coeffs: np.ndarray, marker_size: float):
        """Per-marker pose (rvecs, tvecs) of shape (N,3) float64 via solvePnP IPPE.

        corners: (N,4,2) as returned by detect().
        """
        corners = np.ascontiguousarray(corners, dtype=np.float32)
        camera_matrix = np.ascontiguousarray(camera_matrix, dtype=np.float64)
        dist_coeffs = np.ascontiguousarray(dist_coeffs, dtype=np.float64)
        return self._d.estimate_pose(corners, camera_matrix, dist_coeffs,
                                     float(marker_size))


__all__ = ["__version__", "Dict", "DetectionResult", "ArucoDetector"]
