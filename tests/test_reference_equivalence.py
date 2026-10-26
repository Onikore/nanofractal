"""The optimised detector must return exactly what the upstream reference
detector (third_party/nanofractal.h, exposed as _reference_detect) returns.

Outer-corner detection is bit-identical by design. The inner-point path
differs on purpose in one respect (FAST runs only around the marker, see
src/fractal_fast.hpp), so for it we only require the same detected markers.
"""

import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf
from nanofractal.bench import generate_scene

CONFIGS = ["FRACTAL_2L_6", "FRACTAL_3L_6", "FRACTAL_4L_6", "FRACTAL_5L_6"]


def _params(**kw):
    p = nf.DetectorParams()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _same(img, config, params):
    ref = _nf._reference_detect(config, -1.0, params, img, False)
    got = nf.FractalDetector(config, params=params).detect(img)
    assert got.ids.tolist() == ref[0].tolist()
    np.testing.assert_array_equal(got.corners, ref[1])
    return got


@pytest.mark.parametrize("scene", ["clean", "perspective", "clutter"])
@pytest.mark.parametrize("size", [(640, 480), (1280, 720), (1920, 1080), (2560, 1440)])
def test_scenes_identical(scene, size):
    img = generate_scene(*size, scene)
    _same(img, "FRACTAL_5L_6", _params())


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", [1, 2])
def test_configs_identical(config, seed):
    img = generate_scene(1280, 720, "perspective", config=config, seed=seed)
    res = _same(img, config, _params())
    assert len(res) > 0


@pytest.mark.parametrize(
    "kw",
    [
        dict(adaptive_block_size=7),
        dict(adaptive_block_size=21),  # > 16x16 window: int32 sums, float rounding
        dict(adaptive_block_size=41),
        dict(adaptive_c=3.5),
        dict(adaptive_c=-2.0),
        dict(min_contour_size=40),
        dict(approx_poly_rate=0.03),
        dict(subpix_win_size=0),
        dict(subpix_win_size=2),
        dict(detection_scale=0.5),
    ],
)
def test_params_identical(kw):
    img = generate_scene(1600, 900, "perspective", seed=3)
    _same(img, "FRACTAL_5L_6", _params(**kw))


def test_bgr_and_odd_sizes_identical():
    for w, h in [(641, 479), (1283, 721), (333, 1001)]:
        g = generate_scene(w, h, "perspective", seed=w)
        bgr = np.ascontiguousarray(np.stack([g, g // 2 + 60, 255 - g], axis=-1))
        _same(g, "FRACTAL_5L_6", _params())
        _same(bgr, "FRACTAL_5L_6", _params())


def test_inner_points_same_markers():
    img = generate_scene(1280, 720, "perspective")
    ref = _nf._reference_detect("FRACTAL_5L_6", 1.0, _params(), img, True)
    got = nf.FractalDetector("FRACTAL_5L_6", marker_size=1.0).detect(img, with_inner_points=True)
    assert got.ids.tolist() == ref[0].tolist()
    np.testing.assert_array_equal(got.corners, ref[1])
    assert got.points_2d.shape[0] > 0
