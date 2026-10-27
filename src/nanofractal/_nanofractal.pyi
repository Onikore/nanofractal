from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray

# ── Module-level constants & functions ──────────────────────────────────────
__version__: str

def _opencv_version() -> str: ...
def set_num_threads(n: int) -> None: ...
def _tracer_matches_findcontours(
    image: NDArray[numpy.uint8], min_size: int, band_rows: int, /
) -> tuple[bool, int]: ...
def _area_downscale_diff(
    image: NDArray[numpy.uint8], f: float, threads: int, /
) -> tuple[float, int, bool]: ...
def get_num_threads() -> int: ...
def _fractal_markers(
    config: str, /
) -> list[tuple[int, NDArray[numpy.float32], NDArray[numpy.uint8]]]: ...
def _fractal_external_image8(config: str, /) -> NDArray[numpy.uint8]: ...
def _fractal_external_id(config: str, /) -> int: ...

# ── Classes ──────────────────────────────────────────────────────────────────
class DetectorParams:
    def __init__(self) -> None: ...
    @property
    def min_contour_size(self) -> int:
        """-1 = detector default (120)"""

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
        """corner sub-pixel half-window (-1 = default 4, 0 = off)"""

    @subpix_win_size.setter
    def subpix_win_size(self, arg: int, /) -> None: ...
    @property
    def kfilter_min_dist(self) -> float:
        """min distance (px) between FAST keypoints (default 10)"""

    @kfilter_min_dist.setter
    def kfilter_min_dist(self, arg: float, /) -> None: ...
    @property
    def detection_scale(self) -> float:
        """
        downscale factor for the detection stage (1.0 = full res; 0.5 ~= 4x faster; corners refined at full res)
        """

    @detection_scale.setter
    def detection_scale(self, arg: float, /) -> None: ...
    def __repr__(self) -> str: ...

class FractalDetector:
    def __init__(self, config: str, marker_size: float, params: DetectorParams = ...) -> None: ...
    @property
    def params(self) -> DetectorParams: ...
    @params.setter
    def params(self, arg: DetectorParams, /) -> None: ...
    def detect(self, image: NDArray) -> tuple: ...
    def detect_full(self, image: NDArray) -> tuple: ...
    def diagnose(
        self, image: NDArray, rx: int = -1, ry: int = -1, rw: int = -1, rh: int = -1
    ) -> tuple: ...
    def detect_batch(self, images: Sequence[NDArray], num_threads: int = 0) -> list[object]: ...
    def estimate_pose(
        self,
        points_2d: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
        points_3d: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
        ids: Annotated[NDArray[numpy.int32], dict(order="C", device="cpu", writable=False)],
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
