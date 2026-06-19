from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from . import _nanofractal as _nf

__version__ = _nf.__version__
# Diagnostic hook (which OpenCV the extension is linked against). Kept as a
# private module attribute — intentionally NOT part of the public __all__.
_opencv_version = _nf._opencv_version

# Re-export the C++ DetectorParams class directly — it is a nanobind type with
# read/write attributes and a helpful __repr__. Python users can subclass it or
# just set fields on an instance.
DetectorParams = _nf.DetectorParams


_ALLOWED_D_LENGTHS = frozenset({4, 5, 8, 12, 14})


def _validate_camera(K: np.ndarray, D: np.ndarray) -> None:
    """Validate converted (float64, contiguous) camera-intrinsics arrays.

    Parameters
    ----------
    K : np.ndarray
        Camera matrix — must be float64, shape (3, 3), all finite.
    D : np.ndarray
        Distortion coefficients — must be float64, 1-D, length in
        {4, 5, 8, 12, 14} (OpenCV-supported counts), all finite.

    Raises
    ------
    ValueError
        With a descriptive message on any violation.
    """
    # --- camera matrix shape ---
    if K.ndim != 2 or K.shape != (3, 3):
        raise ValueError(f"camera_matrix must have shape (3, 3), got {K.shape}")
    # --- distortion length ---
    if D.ndim != 1:
        raise ValueError(f"dist_coeffs must be 1-D, got shape {D.shape}")
    if len(D) not in _ALLOWED_D_LENGTHS:
        raise ValueError(
            f"dist_coeffs length must be one of {sorted(_ALLOWED_D_LENGTHS)}, got {len(D)}"
        )
    # --- finiteness ---
    if not np.all(np.isfinite(K)):
        raise ValueError("camera_matrix must contain only finite values (got NaN or inf)")
    if not np.all(np.isfinite(D)):
        raise ValueError("dist_coeffs must contain only finite values (got NaN or inf)")


def _validate_params(p: "DetectorParams") -> None:
    """Validate DetectorParams fields at detect time and raise ValueError on bad values.

    Called at the start of every detect / detect_batch entry point so that
    post-construction mutations (e.g. ``det.params.adaptive_block_size = 4``)
    are still caught before OpenCV's internal assertion fires.
    """
    abs_ = p.adaptive_block_size
    if abs_ != -1 and (abs_ < 3 or abs_ % 2 == 0):
        raise ValueError(
            f"adaptive_block_size must be -1 (auto) or an odd integer >= 3, got {abs_}"
        )

    mcs = p.min_contour_size
    if mcs != -1 and mcs <= 0:
        raise ValueError(f"min_contour_size must be -1 or > 0, got {mcs}")

    apr = p.approx_poly_rate
    if not math.isfinite(apr) or apr <= 0:
        raise ValueError(f"approx_poly_rate must be finite and > 0, got {apr}")

    ac = p.adaptive_c
    if not math.isfinite(ac):
        raise ValueError(f"adaptive_c must be finite, got {ac}")

    sws = p.subpix_win_size
    if sws < -1:
        raise ValueError(f"subpix_win_size must be >= -1, got {sws}")

    kmd = p.kfilter_min_dist
    if not math.isfinite(kmd) or kmd < 0:
        raise ValueError(f"kfilter_min_dist must be finite and >= 0, got {kmd}")

    ds = p.detection_scale
    if not math.isfinite(ds):
        raise ValueError(f"detection_scale must be finite, got {ds}")


def _validate_roi(roi, image: np.ndarray) -> tuple[int, int, int, int]:
    """Validate an ROI tuple against an image and return ``(rx, ry, rw, rh)``.

    Parameters
    ----------
    roi : sequence of 4 ints
        ``(x, y, w, h)`` region of interest.
    image : np.ndarray
        The image the ROI will be applied to; only ``shape[:2]`` is used.

    Returns
    -------
    tuple[int, int, int, int]
        Validated ``(rx, ry, rw, rh)`` as Python ints.

    Raises
    ------
    ValueError
        With a descriptive message on any violation.
    """
    if not hasattr(roi, "__len__") or len(roi) != 4:
        raise ValueError(f"roi must be a length-4 tuple of ints (x, y, w, h), got {roi!r}")
    try:
        rx, ry, rw, rh = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"roi elements must be integers: {exc}") from exc
    if rw <= 0:
        raise ValueError(f"roi width must be > 0, got w={rw}")
    if rh <= 0:
        raise ValueError(f"roi height must be > 0, got h={rh}")
    if rx < 0:
        raise ValueError(f"roi x must be >= 0, got x={rx}")
    if ry < 0:
        raise ValueError(f"roi y must be >= 0, got y={ry}")
    H, W = image.shape[:2]
    if rx + rw > W:
        raise ValueError(f"roi x+w ({rx}+{rw}={rx + rw}) exceeds image width ({W})")
    if ry + rh > H:
        raise ValueError(f"roi y+h ({ry}+{rh}={ry + rh}) exceeds image height ({H})")
    return rx, ry, rw, rh


def _draw_target(image: np.ndarray, inplace: bool) -> np.ndarray:
    """Resolve the array to draw onto for the detectors' ``draw`` methods.

    ``inplace=True`` requires a writable image and returns it unchanged (drawing
    happens directly on the caller's buffer). ``inplace=False`` returns a fresh
    contiguous copy, so a read-only input is fine and the original is preserved.
    """
    if inplace:
        if not image.flags["WRITEABLE"]:
            raise ValueError("image must be writable (or pass inplace=False to draw on a copy)")
        return image
    return np.ascontiguousarray(image).copy()


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


@dataclass(repr=False)
class DetectionResult:
    ids: np.ndarray  # int32 (N,)
    corners: np.ndarray  # float32 (N, 4, 2)
    # points_2d / points_3d are populated only by FractalDetector.detect(
    # ..., with_inner_points=True); they stay None for ArUco and plain fractal
    # detection.
    points_2d: np.ndarray | None = None  # float32 (M, 2)
    points_3d: np.ndarray | None = None  # float32 (M, 3)

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self):
        """Yield ``(marker_id: int, corners: float32 (4,2))`` per marker."""
        for mid, c in zip(self.ids, self.corners):
            yield int(mid), c

    def __repr__(self) -> str:
        n = len(self)
        ids_list = self.ids.tolist()
        _MAX_SHOW = 8
        if n > _MAX_SHOW:
            shown = ids_list[:_MAX_SHOW]
            ids_str = f"[{', '.join(str(i) for i in shown)}, ...]"
        else:
            ids_str = str(ids_list)
        return f"DetectionResult(n={n}, ids={ids_str})"


class ArucoDetector:
    def __init__(
        self,
        dictionary: Dict = Dict.ARUCO_MIP_36h12,
        max_attempts: int = 1,
        params: "DetectorParams | None" = None,
    ) -> None:
        self.dictionary = Dict(dictionary)
        p = params if params is not None else _nf.DetectorParams()
        self._d = _nf.ArucoDetector(int(dictionary), int(max_attempts), p)

    @property
    def max_attempts(self) -> int:
        return self._d.max_attempts

    @property
    def params(self) -> "DetectorParams":
        return self._d.params

    @params.setter
    def params(self, value: "DetectorParams") -> None:
        self._d.params = value

    def __repr__(self) -> str:
        return f"ArucoDetector(dictionary={self.dictionary.name}, max_attempts={self.max_attempts})"

    def detect(
        self,
        image: np.ndarray,
        roi: "tuple[int, int, int, int] | None" = None,
    ) -> DetectionResult:
        _validate_params(self._d.params)
        if roi is not None:
            rx, ry, rw, rh = _validate_roi(roi, image)
            ids, corners = self._d.detect(image, rx, ry, rw, rh)
        else:
            ids, corners = self._d.detect(image)
        return DetectionResult(ids=ids, corners=corners)

    def detect_batch(
        self,
        images,
        num_threads: int = 0,
        roi: "tuple[int, int, int, int] | None" = None,
    ) -> list[DetectionResult]:
        """Detect markers in many images in parallel (0 = use all cores)."""
        _validate_params(self._d.params)
        imgs = list(images)
        if roi is not None:
            for img in imgs:
                _validate_roi(roi, img)
            rx, ry, rw, rh = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            results = self._d.detect_batch(imgs, int(num_threads), rx, ry, rw, rh)
        else:
            results = self._d.detect_batch(imgs, int(num_threads))
        return [DetectionResult(ids=i, corners=c) for i, c in results]

    def estimate_pose(
        self,
        corners: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_size: float,
        return_reproj: bool = False,
        fisheye: bool = False,
    ):
        """Per-marker pose via solvePnP IPPE_SQUARE.

        corners: float32 (N,4,2) as returned by detect().
        camera_matrix: float64 (3,3) intrinsic matrix (must be finite).
        dist_coeffs: float64 (K,) distortion coefficients.
        marker_size: physical side length of the marker in the same unit as tvecs.
        return_reproj: when False (default) returns (rvecs, tvecs), each float64
            (N,3) — identical to previous behaviour.  When True returns
            (rvecs, tvecs, reproj_errs) where reproj_errs is float64 (N,) with the
            per-marker RMS reprojection error in pixels.
        fisheye: when True, treat (camera_matrix, dist_coeffs) as the OpenCV
            fisheye model — dist_coeffs must be exactly 4 (k1,k2,k3,k4). Corners
            are undistorted through the fisheye model before solving, and the
            reprojection error (if requested) is measured against the original
            observations with fisheye projection.
        """
        corners = np.ascontiguousarray(corners, dtype=np.float32)
        camera_matrix = np.ascontiguousarray(camera_matrix, dtype=np.float64)
        dist_coeffs = np.ascontiguousarray(dist_coeffs, dtype=np.float64)
        _validate_camera(camera_matrix, dist_coeffs)
        if fisheye and dist_coeffs.shape[0] != 4:
            raise ValueError(
                "fisheye=True requires exactly 4 distortion coefficients "
                f"(k1,k2,k3,k4), got {dist_coeffs.shape[0]}"
            )
        return self._d.estimate_pose(
            corners,
            camera_matrix,
            dist_coeffs,
            float(marker_size),
            bool(return_reproj),
            bool(fisheye),
        )

    def draw(
        self,
        image: np.ndarray,
        result: DetectionResult,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        rvecs: np.ndarray | None = None,
        tvecs: np.ndarray | None = None,
        marker_size: float | None = None,
        axis_length: float | None = None,
        inplace: bool = True,
    ) -> np.ndarray:
        """Draw marker outlines (+ ids) for every detected marker; with poses
        (``camera_matrix``, ``dist_coeffs``, ``rvecs``, ``tvecs`` as returned by
        :meth:`estimate_pose`, each ``(N,3)``) also draw a frame-axis triad per
        marker.

        ``inplace=True`` (default) draws onto ``image`` and returns it (``image``
        must be writable, contiguous uint8, BGR for colour); ``inplace=False``
        draws on a copy and returns it. ``axis_length`` defaults to half
        ``marker_size`` when given, else ``1.0``.
        """
        target = _draw_target(image, inplace)
        corners = np.ascontiguousarray(result.corners, dtype=np.float32)
        ids = np.ascontiguousarray(result.ids, dtype=np.int32)
        axes = all(v is not None for v in (camera_matrix, dist_coeffs, rvecs, tvecs))
        if axes:
            cam = np.ascontiguousarray(camera_matrix, dtype=np.float64)
            dist = np.ascontiguousarray(dist_coeffs, dtype=np.float64)
            rv = np.ascontiguousarray(np.asarray(rvecs, dtype=np.float64).reshape(-1, 3))
            tv = np.ascontiguousarray(np.asarray(tvecs, dtype=np.float64).reshape(-1, 3))
            if axis_length is None:
                axis_length = marker_size * 0.5 if marker_size and marker_size > 0 else 1.0
        else:
            cam = dist = np.zeros(0, dtype=np.float64)
            rv = tv = np.zeros((0, 3), dtype=np.float64)
            axis_length = 0.0
        self._d.draw(target, corners, ids, bool(axes), cam, dist, rv, tv, float(axis_length))
        return target


_FRACTAL_CONFIGS = {"FRACTAL_2L_6", "FRACTAL_3L_6", "FRACTAL_4L_6", "FRACTAL_5L_6"}


class FractalDetector:
    def __init__(
        self, config: str, marker_size: float = -1.0, params: "DetectorParams | None" = None
    ) -> None:
        if config not in _FRACTAL_CONFIGS:
            raise ValueError(f"invalid config {config!r}; use one of {sorted(_FRACTAL_CONFIGS)}")
        self.config = config
        self.marker_size = float(marker_size)
        p = params if params is not None else _nf.DetectorParams()
        self._d = _nf.FractalDetector(config, float(marker_size), p)

    @property
    def params(self) -> "DetectorParams":
        return self._d.params

    @params.setter
    def params(self, value: "DetectorParams") -> None:
        self._d.params = value

    def __repr__(self) -> str:
        return f"FractalDetector(config={self.config!r}, marker_size={self.marker_size})"

    def detect(
        self,
        image: np.ndarray,
        with_inner_points: bool = False,
        roi: "tuple[int, int, int, int] | None" = None,
    ) -> DetectionResult:
        _validate_params(self._d.params)
        if roi is not None:
            rx, ry, rw, rh = _validate_roi(roi, image)
            if with_inner_points:
                ids, corners, p2d, p3d = self._d.detect_full(image, rx, ry, rw, rh)
                return DetectionResult(ids=ids, corners=corners, points_2d=p2d, points_3d=p3d)
            ids, corners = self._d.detect(image, rx, ry, rw, rh)
        else:
            if with_inner_points:
                ids, corners, p2d, p3d = self._d.detect_full(image)
                return DetectionResult(ids=ids, corners=corners, points_2d=p2d, points_3d=p3d)
            ids, corners = self._d.detect(image)
        return DetectionResult(ids=ids, corners=corners)

    def detect_batch(
        self,
        images,
        num_threads: int = 0,
        roi: "tuple[int, int, int, int] | None" = None,
    ) -> list[DetectionResult]:
        """Detect fractal markers in many images in parallel (0 = all cores)."""
        _validate_params(self._d.params)
        imgs = list(images)
        if roi is not None:
            for img in imgs:
                _validate_roi(roi, img)
            rx, ry, rw, rh = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            results = self._d.detect_batch(imgs, int(num_threads), rx, ry, rw, rh)
        else:
            results = self._d.detect_batch(imgs, int(num_threads))
        return [DetectionResult(ids=i, corners=c) for i, c in results]

    def estimate_pose(
        self,
        result: DetectionResult,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        fisheye: bool = False,
    ):
        """Pose of the fractal marker as ``(rvec, tvec, reproj_err)`` or ``None``.

        Uses the inner+outer correspondences (``result.points_2d``/``points_3d``
        from ``detect(..., with_inner_points=True)``) when there are >= 4 points
        -- accurate and occlusion-robust -- otherwise falls back to the 4 outer
        corners with a square of side ``marker_size``. Returns ``None`` when no
        marker is visible. ``rvec``/``tvec`` are float64 ``(3,)``; ``reproj_err``
        is the RMS reprojection error in pixels (use it to gate noisy poses).

        ``fisheye=True`` treats (camera_matrix, dist_coeffs) as the OpenCV fisheye
        model (dist_coeffs must be exactly 4: k1,k2,k3,k4); points are undistorted
        through the fisheye model before solving and the reprojection error is
        measured against the original observations with fisheye projection.
        """
        cam = np.ascontiguousarray(camera_matrix, dtype=np.float64)
        dist = np.ascontiguousarray(dist_coeffs, dtype=np.float64)
        _validate_camera(cam, dist)
        if fisheye and dist.shape[0] != 4:
            raise ValueError(
                "fisheye=True requires exactly 4 distortion coefficients "
                f"(k1,k2,k3,k4), got {dist.shape[0]}"
            )
        p2d = result.points_2d
        p3d = result.points_3d
        p2d = (
            np.ascontiguousarray(p2d, dtype=np.float32)
            if p2d is not None
            else np.zeros((0, 2), dtype=np.float32)
        )
        p3d = (
            np.ascontiguousarray(p3d, dtype=np.float32)
            if p3d is not None
            else np.zeros((0, 3), dtype=np.float32)
        )
        corners = np.ascontiguousarray(result.corners, dtype=np.float32)
        return self._d.estimate_pose(p2d, p3d, corners, cam, dist, bool(fisheye))

    def draw(
        self,
        image: np.ndarray,
        result: DetectionResult,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        rvec: np.ndarray | None = None,
        tvec: np.ndarray | None = None,
        axis_length: float | None = None,
        inplace: bool = True,
    ) -> np.ndarray:
        """Draw marker outlines (+ ids); with a pose (``camera_matrix``,
        ``dist_coeffs``, ``rvec``, ``tvec``) also draw the frame axes.

        ``inplace=True`` (default) draws onto ``image`` directly and returns it;
        ``image`` must then be a writable contiguous uint8 array (BGR for colour).
        ``inplace=False`` draws on a copy and returns the new array, leaving the
        input untouched (and accepting a read-only input). ``axis_length``
        defaults to half ``marker_size``.
        """
        target = _draw_target(image, inplace)
        corners = np.ascontiguousarray(result.corners, dtype=np.float32)
        ids = np.ascontiguousarray(result.ids, dtype=np.int32)
        axes = all(v is not None for v in (camera_matrix, dist_coeffs, rvec, tvec))
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
        self._d.draw(target, corners, ids, bool(axes), cam, dist, rv, tv, float(axis_length))
        return target


def refine_corners(
    image: np.ndarray,
    corners: np.ndarray,
    win_size: int = 5,
) -> np.ndarray:
    """Refine detected corner positions to sub-pixel accuracy.

    Wraps ``cv::cornerSubPix`` with a ``(win_size × win_size)`` search window
    and stopping criteria ``EPS + MAX_ITER (30 iters, 0.01 px)``.

    Parameters
    ----------
    image : np.ndarray
        Grayscale ``(H, W)`` or BGR ``(H, W, 3)`` uint8 image. The function
        converts BGR to grayscale internally.
    corners : np.ndarray
        Detected corners, ``float32 (N, 4, 2)``, as returned by
        :meth:`ArucoDetector.detect` or :meth:`FractalDetector.detect`.
    win_size : int, optional
        Half-size of the search window (must be >= 1). Default is 5.

    Returns
    -------
    np.ndarray
        Refined corners, ``float32 (N, 4, 2)``.

    Raises
    ------
    ValueError
        If ``win_size < 1``.
    """
    if win_size < 1:
        raise ValueError(f"win_size must be >= 1, got {win_size}")
    image = np.ascontiguousarray(image)
    corners = np.ascontiguousarray(corners, dtype=np.float32)
    return np.asarray(_nf._refine_corners(image, corners, int(win_size)))


def to_opencv(
    result: DetectionResult,
) -> "tuple[list[np.ndarray], np.ndarray]":
    """Convert a :class:`DetectionResult` to the ``cv2.aruco.detectMarkers`` format.

    Parameters
    ----------
    result : DetectionResult
        Detection result from :meth:`ArucoDetector.detect` or similar.

    Returns
    -------
    corners : list[np.ndarray]
        One ``float32 (1, 4, 2)`` array per detected marker — exactly the
        format that ``cv2.aruco.detectMarkers`` returns for its *corners*
        output.
    ids : np.ndarray
        ``int32 (N, 1)`` array of marker ids.  For an empty result returns
        ``np.zeros((0, 1), np.int32)`` rather than ``None`` (unlike cv2).
    """
    n = len(result)
    if n == 0:
        return [], np.zeros((0, 1), dtype=np.int32)
    corners = [result.corners[i].reshape(1, 4, 2).astype(np.float32) for i in range(n)]
    ids = result.ids.reshape(-1, 1).astype(np.int32)
    return corners, ids


def from_opencv(corners, ids) -> DetectionResult:
    """Build a :class:`DetectionResult` from ``cv2.aruco.detectMarkers`` output.

    Parameters
    ----------
    corners : sequence of np.ndarray
        Each element is ``(1, 4, 2)`` or ``(4, 2)`` float32, as returned by
        ``cv2.aruco.detectMarkers``.
    ids : np.ndarray or None
        ``int32 (N, 1)`` or ``(N,)`` array of marker ids, or ``None``
        (cv2 returns ``None`` when no markers are found).

    Returns
    -------
    DetectionResult
        ``points_2d`` and ``points_3d`` are always ``None`` (they are not
        encoded in the cv2 format).
    """
    _empty = DetectionResult(
        ids=np.zeros(0, dtype=np.int32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
    )
    if ids is None:
        return _empty
    ids_arr = np.asarray(ids, dtype=np.int32).ravel()
    n = len(ids_arr)
    if n == 0:
        return _empty
    corners_list = [np.asarray(c, dtype=np.float32).reshape(4, 2) for c in corners]
    corners_arr = np.stack(corners_list, axis=0)
    return DetectionResult(ids=ids_arr, corners=corners_arr)


def dict_grid_size(d: Dict) -> int:
    """Full marker grid size in cells including border.

    For example DICT_4X4 (4-bit inner + 2 border cells) returns 6.
    """
    return _nf._dict_grid_size(int(d))


def dict_num_markers(d: Dict) -> int:
    """Number of markers in the dictionary.

    Raises ValueError for unknown dictionary identifiers.
    """
    return _nf._dict_num_markers(int(d))


# generate.py uses a lazy import for Dict to avoid the circular-import cycle
# (this module imports generate; generate would need Dict from this module).
# Safe to import here because Dict is already defined above.
from .generate import generate_aruco, generate_fractal  # noqa: E402
from .smoother import PoseSmoother  # noqa: E402

__all__ = [
    "__version__",
    "Dict",
    "DetectionResult",
    "DetectorParams",
    "ArucoDetector",
    "FractalDetector",
    "dict_grid_size",
    "dict_num_markers",
    "generate_aruco",
    "generate_fractal",
    "PoseSmoother",
    "refine_corners",
    "to_opencv",
    "from_opencv",
]
