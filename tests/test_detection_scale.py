"""detection_scale: downsampled detection still finds the marker, corners close to full-res."""

import numpy as np

import nanofractal as nf
import nanofractal._nanofractal as _nf

CONFIG = "FRACTAL_5L_6"


def _render(cell: int = 60, margin: int = 100) -> np.ndarray:
    """External fractal level on a white canvas. Large cells survive 0.5x downscale."""
    grid = np.asarray(_nf._fractal_external_image8(CONFIG))
    marker = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    h, w = marker.shape
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = marker
    return np.ascontiguousarray(canvas)


def _half_scale_detector():
    params = nf.DetectorParams()
    params.detection_scale = 0.5
    return nf.FractalDetector(CONFIG, params=params)


def test_detection_scale_finds_marker():
    """detection_scale=0.5 must still detect a rendered fractal marker."""
    ext = _nf._fractal_external_id(CONFIG)
    res = _half_scale_detector().detect(_render())
    assert ext in res.ids.tolist(), "marker not found with detection_scale=0.5"


def test_detection_scale_corners_close_to_full_res():
    """Corners from detection_scale=0.5 must be within a few pixels of full-res corners."""
    img = _render()
    ext = _nf._fractal_external_id(CONFIG)
    res_full = nf.FractalDetector(CONFIG).detect(img)
    res_half = _half_scale_detector().detect(img)
    assert ext in res_full.ids.tolist(), "marker not found at full res (test setup issue)"
    assert ext in res_half.ids.tolist(), "marker not found with detection_scale=0.5"

    c_full = res_full.corners[res_full.ids.tolist().index(ext)]
    c_half = res_half.corners[res_half.ids.tolist().index(ext)]
    # ponytail: 8 px tolerance; tighter would require sub-pixel analysis of the renderer
    np.testing.assert_allclose(c_half, c_full, atol=8.0)
