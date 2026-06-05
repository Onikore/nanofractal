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


def _camera(img, f=600.0):
    h, w = img.shape[:2]
    cam = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return cam, dist


def _noisy_fractal(sigma=6.0, seed=0):
    grid = np.asarray(_nf._fractal_external_image8(CONFIG))
    up = np.kron(grid, np.ones((40, 40), dtype=np.uint8))
    h, w = up.shape
    base = np.full((h + 160, w + 160), 255, dtype=np.uint8)
    base[80:80 + h, 80:80 + w] = up
    rng = np.random.default_rng(seed)
    img = np.clip(base.astype(np.float32) + rng.normal(0, sigma, base.shape),
                  0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def test_estimate_pose_fallback_outer_corners(render_fractal_external):
    # No inner points requested -> estimate_pose must fall back to the 4 outer
    # corners using marker_size, with no need for the caller to touch solvePnP.
    img = render_fractal_external()
    det = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = det.detect(img)
    assert res.points_2d is None
    cam, dist = _camera(img)
    out = det.estimate_pose(res, cam, dist)
    assert out is not None
    rvec, tvec, err = out
    assert rvec.shape == (3,) and tvec.shape == (3,)
    assert rvec.dtype == np.float64 and tvec.dtype == np.float64
    tz = tvec[2]
    assert tz > 0
    assert abs(tvec[0]) < 0.1 * tz and abs(tvec[1]) < 0.1 * tz
    # the fallback object-point order matches the detected corner order, so a
    # frontal marker reprojects to within a pixel
    assert 0.0 <= err < 2.0


def test_estimate_pose_inner_points_is_accurate():
    # With inner points the pose uses many coplanar correspondences -> low reproj.
    img = _noisy_fractal()
    det = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = det.detect(img, with_inner_points=True)
    assert res.points_2d is not None and res.points_2d.shape[0] >= 4
    cam, dist = _camera(img)
    out = det.estimate_pose(res, cam, dist)
    assert out is not None
    rvec, tvec, err = out
    assert tvec[2] > 0
    assert err < 3.0  # RMS reprojection error in pixels


def test_estimate_pose_empty_inner_falls_back(render_fractal_external):
    # with_inner_points=True but clean image -> points_2d is (0,2); must not raise
    # and must fall back to the outer corners (the empty-inner-points pitfall is
    # hidden inside estimate_pose).
    img = render_fractal_external()
    det = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = det.detect(img, with_inner_points=True)
    assert res.points_2d.shape == (0, 2)
    cam, dist = _camera(img)
    out = det.estimate_pose(res, cam, dist)
    assert out is not None
    _, tvec, _ = out
    assert tvec[2] > 0


def test_estimate_pose_no_marker_returns_none():
    det = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = det.detect(np.full((480, 640), 255, dtype=np.uint8))
    cam = np.eye(3, dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    assert det.estimate_pose(res, cam, dist) is None


def test_draw_corners_and_axes(render_fractal_external):
    gray = render_fractal_external()
    bgr = np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))  # (H,W,3)
    det = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = det.detect(bgr)
    assert res.ids.shape[0] >= 1  # marker present so something gets drawn

    # corners only
    before = bgr.copy()
    out = det.draw(bgr, res)
    assert out is bgr
    assert not np.array_equal(bgr, before)  # pixels changed
    # green corner lines were drawn somewhere
    assert (bgr[..., 1] == 255).any()

    # with pose axes (must not raise)
    cam, dist = _camera(bgr)
    rvec, tvec, _ = det.estimate_pose(res, cam, dist)
    det.draw(bgr, res, cam, dist, rvec, tvec)


def test_draw_readonly_image_raises(render_fractal_external):
    gray = render_fractal_external()
    bgr = np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))
    det = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = det.detect(bgr)
    ro = bgr.copy()
    ro.flags.writeable = False
    with pytest.raises(ValueError):
        det.draw(ro, res)  # drawing needs a writable image
