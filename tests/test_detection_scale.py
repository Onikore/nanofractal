"""detection_scale: downsampled detection still finds the marker, corners close to full-res."""

import numpy as np

import nanofractal as nf
import nanofractal._nanofractal as _nf


def _render_aruco(marker_id: int = 0, cell: int = 60, margin: int = 100) -> np.ndarray:
    """Render an ArUco marker on a white canvas. Larger cell so it survives 0.5x downscale."""
    grid = np.asarray(_nf._aruco_marker_image8(0, marker_id))
    marker = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    h, w = marker.shape
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = marker
    return np.ascontiguousarray(canvas)


def test_detection_scale_finds_marker():
    """detection_scale=0.5 detector must still detect a rendered ArUco marker."""
    img = _render_aruco()
    params = nf.DetectorParams()
    params.detection_scale = 0.5
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10, params=params)
    res = det.detect(img)
    assert 0 in res.ids.tolist(), "marker 0 not found with detection_scale=0.5"


def test_detection_scale_corners_close_to_full_res():
    """Corners from detection_scale=0.5 must be within a few pixels of full-res corners."""
    img = _render_aruco()

    # Full-res detector
    det_full = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res_full = det_full.detect(img)
    assert 0 in res_full.ids.tolist(), "marker not found at full res (test setup issue)"

    # Half-res detection
    params = nf.DetectorParams()
    params.detection_scale = 0.5
    det_half = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10, params=params)
    res_half = det_half.detect(img)
    assert 0 in res_half.ids.tolist(), "marker 0 not found with detection_scale=0.5"

    idx_full = res_full.ids.tolist().index(0)
    idx_half = res_half.ids.tolist().index(0)
    c_full = res_full.corners[idx_full]
    c_half = res_half.corners[idx_half]

    # ponytail: 8 px tolerance; tighter would require sub-pixel analysis of the renderer
    np.testing.assert_allclose(c_half, c_full, atol=8.0)
