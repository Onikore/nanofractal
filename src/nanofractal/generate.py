"""Marker image generation utilities for nanofractal.

Public API
----------
generate_fractal(config, size_px=400, nested=True)
    Render a fractal marker as a grayscale uint8 image. By default every
    nested level is composited (a printable marker); ``nested=False`` renders
    only the external (outermost) level.
"""

from __future__ import annotations

import numpy as np

from . import _nanofractal as _nf

_FRACTAL_CONFIGS = frozenset({"FRACTAL_2L_6", "FRACTAL_3L_6", "FRACTAL_4L_6", "FRACTAL_5L_6"})


def _nn_upscale(grid: np.ndarray, size_px: int) -> np.ndarray:
    """Nearest-neighbour upscale a (total x total) cell grid to (size_px x size_px).

    Pure-numpy implementation: no cv2 dependency.
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


def _render_nested(config: str, size_px: int) -> np.ndarray:
    """Composite every marker level of *config* into one (size_px, size_px) image.

    Each level is a black-bordered bit grid placed at its normalized corners
    ([-1, 1], y up). Levels are painted largest first, so each submarker
    overwrites the masked-out region of its parent. Pixels are sampled at their
    centres (nearest cell), so cell edges are exact at any size.
    """
    markers = _nf._fractal_markers(config)
    # largest first: children overwrite the parent's masked region
    markers = sorted(markers, key=lambda m: -(np.ptp(np.asarray(m[1])[:, 0])))
    img = np.full((size_px, size_px), 255, dtype=np.uint8)
    # pixel centres in normalized coords: x right, y up
    centres = (np.arange(size_px, dtype=np.float64) + 0.5) / size_px * 2.0 - 1.0
    xs = centres
    ys = -centres
    for _id, corners, bits in markers:
        c = np.asarray(corners, dtype=np.float64)
        bits = np.asarray(bits)
        x0, x1 = c[:, 0].min(), c[:, 0].max()
        y0, y1 = c[:, 1].min(), c[:, 1].max()
        n = bits.shape[0] + 2  # cells per side incl. the black border
        cols = np.nonzero((xs >= x0) & (xs < x1))[0]
        rows = np.nonzero((ys > y0) & (ys <= y1))[0]
        if cols.size == 0 or rows.size == 0:
            continue
        cx = np.minimum(((xs[cols] - x0) / (x1 - x0) * n).astype(np.intp), n - 1)
        cy = np.minimum(((y1 - ys[rows]) / (y1 - y0) * n).astype(np.intp), n - 1)
        grid = np.zeros((n, n), dtype=np.uint8)
        grid[1:-1, 1:-1] = np.where(bits > 0, 255, 0)
        img[np.ix_(rows, cols)] = grid[np.ix_(cy, cx)]
    return np.ascontiguousarray(img)


def generate_fractal(config: str, size_px: int = 400, nested: bool = True) -> np.ndarray:
    """Render a fractal marker as a square grayscale image.

    Parameters
    ----------
    config : str
        Fractal configuration string, one of
        ``"FRACTAL_2L_6"``, ``"FRACTAL_3L_6"``, ``"FRACTAL_4L_6"``,
        ``"FRACTAL_5L_6"``.
    size_px : int
        Side length of the returned image in pixels (default 400). For a
        crisp print choose a multiple of the smallest level's cell count.
    nested : bool
        ``True`` (default) composites every nested level — the full,
        printable fractal marker. ``False`` renders only the external
        (outermost) level.

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(size_px, size_px)``, values 0/255,
        C-contiguous. The image spans the marker's outer black border exactly
        (add your own white quiet zone around it).

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

    if nested:
        return _render_nested(config, int(size_px))
    grid = np.asarray(_nf._fractal_external_image8(config))
    return _nn_upscale(grid, size_px)
