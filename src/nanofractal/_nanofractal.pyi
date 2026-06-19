from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray

# ── Module-level constants & functions ──────────────────────────────────────
__version__: str

def _opencv_version() -> str: ...
def _aruco_marker_image8(dictionary: int, marker_id: int, /) -> NDArray[numpy.uint8]: ...
def _dict_grid_size(dictionary: int, /) -> int: ...
def _dict_num_markers(dictionary: int, /) -> int: ...
def _fractal_external_image8(config: str, /) -> NDArray[numpy.uint8]: ...
def _fractal_external_id(config: str, /) -> int: ...

# ── Classes ──────────────────────────────────────────────────────────────────
class DetectorParams:
    def __init__(self) -> None: ...
    @property
    def min_contour_size(self) -> int:
        """-1 = detector default (ArUco: 50, Fractal: 120)"""

    @min_contour_size.setter
    def min_contour_size(self, arg: int, /) -> None: ...
    @property
    def adaptive_block_size(self) -> int:
        """-1 = auto-scale with image width"""

    @adaptive_block_size.setter
    def adaptive_block_size(self, arg: int, /) -> None: ...
    @property
    def adaptive_c(self) -> float:
        """constant subtracted from adaptive threshold mean (default 7)"""

    @adaptive_c.setter
    def adaptive_c(self, arg: float, /) -> None: ...
    @property
    def approx_poly_rate(self) -> float:
        """contour approx epsilon = perimeter * rate (default 0.05)"""

    @approx_poly_rate.setter
    def approx_poly_rate(self, arg: float, /) -> None: ...
    @property
    def subpix_win_size(self) -> int:
        """corner sub-pixel half-window, Fractal only (-1 = default 4, 0 = off)"""

    @subpix_win_size.setter
    def subpix_win_size(self, arg: int, /) -> None: ...
    @property
    def kfilter_min_dist(self) -> float:
        """min distance (px) between FAST keypoints, Fractal only (default 10)"""

    @kfilter_min_dist.setter
    def kfilter_min_dist(self, arg: float, /) -> None: ...
    @property
    def detection_scale(self) -> float:
        """
        downscale factor for the detection stage, both detectors (1.0 = full res; 0.5 ~= 4x faster; corners refined at full res)
        """

    @detection_scale.setter
    def detection_scale(self, arg: float, /) -> None: ...
    def __repr__(self) -> str: ...

class ArucoDetector:
    def __init__(
        self, dictionary: int, max_attempts: int, params: DetectorParams = ...
    ) -> None: ...
    @property
    def max_attempts(self) -> int: ...
    @property
    def dictionary(self) -> int: ...
    @property
    def params(self) -> DetectorParams: ...
    @params.setter
    def params(self, arg: DetectorParams, /) -> None: ...
    def detect(self, image: NDArray) -> tuple: ...
    def estimate_pose(
        self,
        corners: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
        camera_matrix: Annotated[
            NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)
        ],
        dist_coeffs: Annotated[
            NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)
        ],
        marker_size: float,
        return_reproj: bool = False,
        fisheye: bool = False,
    ) -> tuple: ...
    def detect_batch(self, images: Sequence[NDArray], num_threads: int = 0) -> list[object]: ...
    def draw(
        self,
        image: NDArray,
        corners: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
        ids: Annotated[NDArray[numpy.int32], dict(order="C", device="cpu", writable=False)],
        axes: bool,
        cam: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        dist: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        rvecs: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        tvecs: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        axis_length: float,
    ) -> None: ...

class FractalDetector:
    def __init__(self, config: str, marker_size: float, params: DetectorParams = ...) -> None: ...
    @property
    def params(self) -> DetectorParams: ...
    @params.setter
    def params(self, arg: DetectorParams, /) -> None: ...
    def detect(self, image: NDArray) -> tuple: ...
    def detect_full(self, image: NDArray) -> tuple: ...
    def detect_batch(self, images: Sequence[NDArray], num_threads: int = 0) -> list[object]: ...
    def estimate_pose(
        self,
        points_2d: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
        points_3d: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
        corners: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
        camera_matrix: Annotated[
            NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)
        ],
        dist_coeffs: Annotated[
            NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)
        ],
        fisheye: bool = False,
    ) -> object: ...
    def draw(
        self,
        image: NDArray,
        corners: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
        ids: Annotated[NDArray[numpy.int32], dict(order="C", device="cpu", writable=False)],
        axes: bool,
        cam: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        dist: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        rvec: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        tvec: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        axis_length: float,
    ) -> None: ...
