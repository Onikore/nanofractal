import numpy as np
import pytest
import nanofractal as nf
import nanofractal._nanofractal as _nf

CONFIG = "FRACTAL_5L_6"


@pytest.fixture
def render_fractal_external():
    def _render(cell: int = 40, margin: int = 80) -> np.ndarray:
        grid = np.asarray(_nf._fractal_external_image8(CONFIG))
        up = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
        h, w = up.shape
        canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
        canvas[margin:margin + h, margin:margin + w] = up
        return np.ascontiguousarray(canvas)
    return _render


def test_blank_returns_empty():
    det = nf.FractalDetector(CONFIG)
    res = det.detect(np.full((480, 640), 255, dtype=np.uint8))
    assert res.ids.dtype == np.int32 and res.ids.shape == (0,)
    assert res.corners.dtype == np.float32 and res.corners.shape == (0, 4, 2)


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        nf.FractalDetector("NOT_A_CONFIG")


def test_detects_external_marker(render_fractal_external):
    img = render_fractal_external()
    det = nf.FractalDetector(CONFIG)
    res = det.detect(img)
    ext_id = _nf._fractal_external_id(CONFIG)
    assert ext_id in res.ids.tolist()
    idx = res.ids.tolist().index(ext_id)
    assert res.corners[idx].shape == (4, 2)
