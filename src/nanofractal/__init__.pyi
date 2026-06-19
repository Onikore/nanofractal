from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Literal, Sequence, overload

import numpy as np
import numpy.typing as npt

# ── Public API ────────────────────────────────────────────────────────────────
__version__: str

# Private (not in __all__): exposed for bench.py / diagnostics.
# At runtime this is a reference to _nanofractal._opencv_version (nb_func).
def _opencv_version() -> str: ...

# ── DetectorParams (re-exported from the C extension) ────────────────────────
class DetectorParams:
    min_contour_size: int
    adaptive_block_size: int
    adaptive_c: float
    approx_poly_rate: float
    subpix_win_size: int
    kfilter_min_dist: float
    detection_scale: float

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

# ── Dict ─────────────────────────────────────────────────────────────────────
class Dict(IntEnum):
    ARUCO_MIP_36h12 = 0
    APRILTAG_36h11 = 1
    DICT_4X4_50 = 2
    DICT_4X4_100 = 3
    DICT_4X4_250 = 4
    DICT_4X4_1000 = 5
    DICT_5X5_50 = 6
    DICT_5X5_100 = 7
    DICT_5X5_250 = 8
    DICT_5X5_1000 = 9
    DICT_6X6_50 = 10
    DICT_6X6_100 = 11
    DICT_6X6_250 = 12
    DICT_6X6_1000 = 13
    DICT_7X7_50 = 14
    DICT_7X7_100 = 15
    DICT_7X7_250 = 16
    DICT_7X7_1000 = 17

# ── DetectionResult ───────────────────────────────────────────────────────────
@dataclass(repr=False)
class DetectionResult:
    ids: npt.NDArray[np.int32]
    corners: npt.NDArray[np.float32]
    points_2d: npt.NDArray[np.float32] | None = ...
    points_3d: npt.NDArray[np.float32] | None = ...

    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __iter__(self) -> Iterator[tuple[int, npt.NDArray[np.float32]]]: ...
    def __repr__(self) -> str: ...

# ── ArucoDetector ─────────────────────────────────────────────────────────────
class ArucoDetector:
    dictionary: Dict

    def __init__(
        self,
        dictionary: Dict = ...,
        max_attempts: int = ...,
        params: DetectorParams | None = ...,
    ) -> None: ...
    @property
    def max_attempts(self) -> int: ...
    @property
    def params(self) -> DetectorParams: ...
    @params.setter
    def params(self, value: DetectorParams) -> None: ...
    def __repr__(self) -> str: ...
    def detect(
        self,
        image: npt.NDArray[np.uint8],
        roi: tuple[int, int, int, int] | None = ...,
    ) -> DetectionResult: ...
    def detect_batch(
        self,
        images: Sequence[npt.NDArray[np.uint8]],
        num_threads: int = ...,
        roi: tuple[int, int, int, int] | None = ...,
    ) -> list[DetectionResult]: ...
    @overload
    def estimate_pose(
        self,
        corners: npt.NDArray[np.float32],
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
        marker_size: float,
        return_reproj: Literal[False] = ...,
        fisheye: bool = ...,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @overload
    def estimate_pose(
        self,
        corners: npt.NDArray[np.float32],
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
        marker_size: float,
        return_reproj: Literal[True],
        fisheye: bool = ...,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]: ...
    @overload
    def estimate_pose(
        self,
        corners: npt.NDArray[np.float32],
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
        marker_size: float,
        return_reproj: bool,
        fisheye: bool = ...,
    ) -> (
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        | tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ]
    ): ...
    def draw(
        self,
        image: npt.NDArray[np.uint8],
        result: DetectionResult,
        camera_matrix: npt.NDArray[np.float64] | None = ...,
        dist_coeffs: npt.NDArray[np.float64] | None = ...,
        rvecs: npt.NDArray[np.float64] | None = ...,
        tvecs: npt.NDArray[np.float64] | None = ...,
        marker_size: float | None = ...,
        axis_length: float | None = ...,
        inplace: bool = ...,
    ) -> npt.NDArray[np.uint8]: ...

# ── FractalDetector ───────────────────────────────────────────────────────────
class FractalDetector:
    config: str
    marker_size: float

    def __init__(
        self,
        config: str,
        marker_size: float = ...,
        params: DetectorParams | None = ...,
    ) -> None: ...
    @property
    def params(self) -> DetectorParams: ...
    @params.setter
    def params(self, value: DetectorParams) -> None: ...
    def __repr__(self) -> str: ...
    def detect(
        self,
        image: npt.NDArray[np.uint8],
        with_inner_points: bool = ...,
        roi: tuple[int, int, int, int] | None = ...,
    ) -> DetectionResult: ...
    def detect_batch(
        self,
        images: Sequence[npt.NDArray[np.uint8]],
        num_threads: int = ...,
        roi: tuple[int, int, int, int] | None = ...,
    ) -> list[DetectionResult]: ...
    def estimate_pose(
        self,
        result: DetectionResult,
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
        fisheye: bool = ...,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float] | None: ...
    def draw(
        self,
        image: npt.NDArray[np.uint8],
        result: DetectionResult,
        camera_matrix: npt.NDArray[np.float64] | None = ...,
        dist_coeffs: npt.NDArray[np.float64] | None = ...,
        rvec: npt.NDArray[np.float64] | None = ...,
        tvec: npt.NDArray[np.float64] | None = ...,
        axis_length: float | None = ...,
        inplace: bool = ...,
    ) -> npt.NDArray[np.uint8]: ...

# ── PoseSmoother ─────────────────────────────────────────────────────────────
class PoseSmoother:
    def __init__(
        self,
        process_noise: float = ...,
        measurement_noise: float = ...,
        mode: str = ...,
    ) -> None: ...
    def update(
        self,
        rvec: npt.ArrayLike,
        tvec: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def reset(self) -> None: ...

# ── Generator functions ───────────────────────────────────────────────────────
def generate_aruco(
    marker_id: int,
    size_px: int = ...,
    dictionary: Dict | int | None = ...,
    border_bits: int = ...,
) -> npt.NDArray[np.uint8]: ...
def generate_fractal(
    config: str,
    size_px: int = ...,
) -> npt.NDArray[np.uint8]: ...

# ── Dict utility functions ────────────────────────────────────────────────────
def dict_grid_size(d: Dict) -> int: ...
def dict_num_markers(d: Dict) -> int: ...

# ── B2: sub-pixel corner refinement ──────────────────────────────────────────
def refine_corners(
    image: npt.NDArray[np.uint8],
    corners: npt.NDArray[np.float32],
    win_size: int = ...,
) -> npt.NDArray[np.float32]: ...

# ── B3: OpenCV interop ────────────────────────────────────────────────────────
def to_opencv(
    result: DetectionResult,
) -> tuple[list[npt.NDArray[np.float32]], npt.NDArray[np.int32]]: ...
def from_opencv(
    corners: Sequence[npt.NDArray[np.float32]],
    ids: npt.NDArray[np.int32] | None,
) -> DetectionResult: ...

__all__ = [
    "__version__",
    "Dict",
    "DetectionResult",
    "DetectorParams",
    "ArucoDetector",
    "FractalDetector",
    "PoseSmoother",
    "dict_grid_size",
    "dict_num_markers",
    "generate_aruco",
    "generate_fractal",
    "refine_corners",
    "to_opencv",
    "from_opencv",
]
