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


def test_detect_empty_frame_raises():
    det = nf.FractalDetector(CONFIG)
    with pytest.raises(ValueError):
        det.detect(np.zeros((0, 640), dtype=np.uint8))


def test_detects_external_marker(render_fractal_external):
    img = render_fractal_external()
    det = nf.FractalDetector(CONFIG)
    res = det.detect(img)
    ext_id = _nf._fractal_external_id(CONFIG)
    assert ext_id in res.ids.tolist()
    idx = res.ids.tolist().index(ext_id)
    assert res.corners[idx].shape == (4, 2)


def test_detect_without_inner_points_has_none(render_fractal_external):
    img = render_fractal_external()
    det = nf.FractalDetector(CONFIG)
    res = det.detect(img)
    assert res.points_2d is None and res.points_3d is None


def test_detect_with_inner_points_empty_is_safe(render_fractal_external):
    # Regression: on a clean synthetic marker FAST finds no keypoints, so the
    # inner-point path must return empty (0,2)/(0,3) arrays rather than segfault
    # (the upstream header dereferenced kpoints[0] on an empty vector).
    img = render_fractal_external()
    det = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = det.detect(img, with_inner_points=True)
    assert _nf._fractal_external_id(CONFIG) in res.ids.tolist()
    assert res.points_2d.shape == (0, 2) and res.points_3d.shape == (0, 3)
    assert res.points_2d.dtype == np.float32 and res.points_3d.dtype == np.float32


def test_detect_with_inner_points_nonempty():
    # With mild noise FAST finds corners, so inner-point correspondences are
    # populated -- exercises the non-empty marshaling of points_2d/points_3d.
    grid = np.asarray(_nf._fractal_external_image8(CONFIG))
    up = np.kron(grid, np.ones((40, 40), dtype=np.uint8))
    h, w = up.shape
    base = np.full((h + 160, w + 160), 255, dtype=np.uint8)
    base[80:80 + h, 80:80 + w] = up
    rng = np.random.default_rng(0)
    img = np.clip(base.astype(np.float32) + rng.normal(0, 6, base.shape),
                  0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)

    det = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = det.detect(img, with_inner_points=True)
    assert _nf._fractal_external_id(CONFIG) in res.ids.tolist()
    m = res.points_2d.shape[0]
    assert m > 0
    assert res.points_2d.shape == (m, 2) and res.points_3d.shape == (m, 3)
    assert res.points_2d.dtype == np.float32 and res.points_3d.dtype == np.float32
    assert (res.points_2d[:, 0] >= 0).all() and (res.points_2d[:, 0] < img.shape[1]).all()
    assert (res.points_2d[:, 1] >= 0).all() and (res.points_2d[:, 1] < img.shape[0]).all()
    # object points are within the marker (marker_size=0.85 -> |coord| <= ~0.43m)
    assert np.abs(res.points_3d).max() < 1.0
    assert (res.points_3d[:, 2] == 0).all()  # planar marker
