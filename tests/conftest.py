import numpy as np
import pytest

import nanofractal._nanofractal as _nf


def _upscale(grid8: np.ndarray, cell: int) -> np.ndarray:
    """Nearest-neighbor upscale an (8,8) grid to (8*cell, 8*cell)."""
    return np.kron(grid8, np.ones((cell, cell), dtype=np.uint8))


@pytest.fixture
def render_aruco():
    def _render(
        marker_id: int, dictionary: int = 0, cell: int = 40, margin: int = 60
    ) -> np.ndarray:
        grid = np.asarray(_nf._aruco_marker_image8(dictionary, marker_id))
        marker = _upscale(grid, cell)
        h, w = marker.shape
        canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
        canvas[margin : margin + h, margin : margin + w] = marker
        return np.ascontiguousarray(canvas)

    return _render
