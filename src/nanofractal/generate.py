"""Marker image generation utilities for nanofractal.

Public API
----------
generate_aruco(marker_id, size_px=200, dictionary=None, border_bits=1)
    Render an ArUco marker as a grayscale uint8 image ready for printing.

generate_fractal(config, size_px=400)
    Render the **external (outermost) fractal marker level** as a grayscale
    uint8 image.

    IMPORTANT CAVEAT: this function renders only the external / outermost
    marker ring that ``FractalDetector`` actually detects. It is NOT the full
    multi-level nested fractal composite image (which would require additional
    library support to compose all levels together). If you need a printable
    nested composite, you must assemble the inner levels separately.
"""

from __future__ import annotations

import numpy as np

from . import _nanofractal as _nf

_FRACTAL_CONFIGS = frozenset({"FRACTAL_2L_6", "FRACTAL_3L_6", "FRACTAL_4L_6", "FRACTAL_5L_6"})


def _nn_upscale(grid: np.ndarray, size_px: int) -> np.ndarray:
    """Nearest-neighbour upscale a (total x total) cell grid to (size_px x size_px).

    Pure-numpy implementation: no cv2 dependency.

    Parameters
    ----------
    grid : np.ndarray
        2-D uint8 array of shape (total, total).
    size_px : int
        Target output dimension (square).

    Returns
    -------
    np.ndarray
        Contiguous uint8 array of shape (size_px, size_px).
    """
    total = grid.shape[0]
    # Map each output pixel to the nearest source cell via integer floor.
    # Clamp to [0, total-1] to handle the edge case when size_px == total.
    iy = np.arange(size_px, dtype=np.intp) * total // size_px
    ix = np.arange(size_px, dtype=np.intp) * total // size_px
    iy = np.clip(iy, 0, total - 1)
    ix = np.clip(ix, 0, total - 1)
    out = grid[np.ix_(iy, ix)]
    return np.ascontiguousarray(out, dtype=np.uint8)


def generate_aruco(
    marker_id: int,
    size_px: int = 200,
    dictionary=None,
    border_bits: int = 1,
) -> np.ndarray:
    """Render an ArUco marker as a square grayscale image.

    Parameters
    ----------
    marker_id : int
        Index of the marker within *dictionary*.
    size_px : int
        Side length of the returned image in pixels (default 200).
    dictionary : Dict or int, optional
        ArUco dictionary to use.  Defaults to ``Dict.ARUCO_MIP_36h12`` when
        ``None``.  Accepts either a ``Dict`` enum member or a plain ``int``.
    border_bits : int
        Width of the black quiet-zone border in marker cells (default 1).
        Minimum value is 1.

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(size_px, size_px)``.  Values are 0
        (black) or 255 (white).  The image is C-contiguous.

    Raises
    ------
    ValueError
        If ``size_px <= 0``, ``border_bits < 1``, or the
        ``marker_id`` / ``dictionary`` combination is invalid (propagated from
        the C++ helper).
    """
    if size_px <= 0:
        raise ValueError(f"size_px must be > 0, got {size_px!r}")
    if border_bits < 1:
        raise ValueError(f"border_bits must be >= 1, got {border_bits!r}")

    # Lazy import avoids a circular-import cycle: __init__ imports us, and we
    # would normally need Dict from __init__.  Using a sentinel + lazy import
    # sidesteps this.
    if dictionary is None:
        from . import Dict

        dictionary = Dict.ARUCO_MIP_36h12

    # _aruco_marker_image8 raises ValueError for unknown dict / out-of-range id.
    grid = np.asarray(_nf._aruco_marker_image8(int(dictionary), int(marker_id)))
    # grid is (gsize x gsize); gsize includes a 1-cell black border already.
    gsize = grid.shape[0]  # noqa: F841

    # Extract the inner bit block (strip the 1-cell border the C++ added).
    inner = grid[1:-1, 1:-1]  # shape (N, N), N = gsize - 2
    N = inner.shape[0]

    # Rebuild with the requested border width.
    total = N + 2 * border_bits
    full = np.zeros((total, total), dtype=np.uint8)  # black background
    full[border_bits : border_bits + N, border_bits : border_bits + N] = inner

    return _nn_upscale(full, size_px)


def generate_fractal(config: str, size_px: int = 400) -> np.ndarray:
    """Render the external (outermost) fractal marker level as a grayscale image.

    This renders the single outermost marker ring that ``FractalDetector``
    detects.

    .. important::

        This function renders **only the external / outermost marker level**
        — i.e. the ArUco-style ring that ``FractalDetector.detect()``
        locates.  It does **not** produce the full multi-level nested fractal
        composite image (which requires compositing all inner levels on top of
        the outer one — additional support not currently in this library).
        Use this image to verify detector round-trips; for a printable
        production target you will need the full composite from the original
        ``fractal_marker_generator`` tool.

    Parameters
    ----------
    config : str
        Fractal configuration string, one of
        ``"FRACTAL_2L_6"``, ``"FRACTAL_3L_6"``, ``"FRACTAL_4L_6"``,
        ``"FRACTAL_5L_6"``.
    size_px : int
        Side length of the returned image in pixels (default 400).

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(size_px, size_px)``.  C-contiguous.

    Raises
    ------
    ValueError
        If *config* is not a recognised fractal configuration string, or if
        ``size_px <= 0``.
    """
    if config not in _FRACTAL_CONFIGS:
        raise ValueError(
            f"invalid fractal config {config!r}; must be one of {sorted(_FRACTAL_CONFIGS)}"
        )
    if size_px <= 0:
        raise ValueError(f"size_px must be > 0, got {size_px!r}")

    grid = np.asarray(_nf._fractal_external_image8(config))
    return _nn_upscale(grid, size_px)
