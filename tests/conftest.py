import numpy as np
import pytest

import nanofractal._nanofractal as _nf


def _upscale(grid: np.ndarray, cell: int) -> np.ndarray:
    """Nearest-neighbor upscale a cell grid by an integer factor."""
    return np.kron(grid, np.ones((cell, cell), dtype=np.uint8))


@pytest.fixture
def render_fractal():
    """Render the external level of a fractal config on a white canvas."""

    def _render(config: str = "FRACTAL_5L_6", cell: int = 40, margin: int = 80) -> np.ndarray:
        grid = np.asarray(_nf._fractal_external_image8(config))
        marker = _upscale(grid, cell)
        h, w = marker.shape
        canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
        canvas[margin : margin + h, margin : margin + w] = marker
        return np.ascontiguousarray(canvas)

    return _render
