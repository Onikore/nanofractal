from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

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
def generate_fractal(
    config: str,
    size_px: int = ...,
    nested: bool = ...,
) -> npt.NDArray[np.uint8]: ...

# ── threading ────────────────────────────────────────────────────────────────
def set_num_threads(n: int) -> None: ...
def get_num_threads() -> int: ...

# ── B2: sub-pixel corner refinement ──────────────────────────────────────────
def refine_corners(
    image: npt.NDArray[np.uint8],
    corners: npt.NDArray[np.float32],
    win_size: int = ...,
) -> npt.NDArray[np.float32]: ...

__all__ = [
    "__version__",
    "DetectionResult",
    "DetectorParams",
    "FractalDetector",
    "PoseSmoother",
    "generate_fractal",
    "refine_corners",
    "set_num_threads",
    "get_num_threads",
]
