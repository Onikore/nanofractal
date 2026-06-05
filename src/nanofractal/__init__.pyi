from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np
import numpy.typing as npt

__version__: str

class Dict(IntEnum):
    ARUCO_MIP_36h12 = 0
    APRILTAG_36h11 = 1

@dataclass
class DetectionResult:
    ids: npt.NDArray[np.int32]
    corners: npt.NDArray[np.float32]
    points_2d: npt.NDArray[np.float32] | None = ...
    points_3d: npt.NDArray[np.float32] | None = ...

class ArucoDetector:
    dictionary: Dict
    def __init__(self, dictionary: Dict = ..., max_attempts: int = ...) -> None: ...
    @property
    def max_attempts(self) -> int: ...
    def detect(self, image: npt.NDArray[np.uint8]) -> DetectionResult: ...
    def detect_batch(
        self, images: Sequence[npt.NDArray[np.uint8]], num_threads: int = ...
    ) -> list[DetectionResult]: ...
    def estimate_pose(
        self,
        corners: npt.NDArray[np.float32],
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
        marker_size: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

class FractalDetector:
    config: str
    marker_size: float
    def __init__(self, config: str, marker_size: float = ...) -> None: ...
    def detect(
        self, image: npt.NDArray[np.uint8], with_inner_points: bool = ...
    ) -> DetectionResult: ...
    def detect_batch(
        self, images: Sequence[npt.NDArray[np.uint8]], num_threads: int = ...
    ) -> list[DetectionResult]: ...
    def estimate_pose(
        self,
        result: DetectionResult,
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float] | None: ...
