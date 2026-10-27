"""The optimised detector must return exactly what the upstream reference
detector (third_party/nanofractal.h, exposed as _reference_detect) returns.

Outer-corner detection is bit-identical by design for every level the
reference finds. Quads the reference reading rejects get a second reading
(refined corners, exact cell sampling: R16), which may add levels the
reference misses -- typically the innermost one at a few pixels per cell;
those must lie inside the marker. The inner-point path differs on purpose in
one respect (FAST runs only around the marker, see src/fractal_fast.hpp), so
for it we only require the same detected markers.
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


def _check_against_reference(got, ref_ids, ref_corners):
    ids = got.ids.tolist()
    assert set(ref_ids.tolist()) <= set(ids)
    for i, mid in enumerate(ref_ids.tolist()):
        np.testing.assert_array_equal(got.corners[ids.index(mid)], ref_corners[i])
    if len(ids):
        outer = got.corners[0]  # ids are sorted: the largest level found
        lo, hi = outer.min(0), outer.max(0)
        for mid, c in zip(ids, got.corners):
            if mid not in ref_ids:  # added by the second reading
                assert np.all(c >= lo) and np.all(c <= hi), (mid, c, outer)


def _same(img, config, params):
    ref = _nf._reference_detect(config, -1.0, params, img, False)
    got = nf.FractalDetector(config, params=params).detect(img)
    _check_against_reference(got, ref[0], ref[1])
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
        dict(detection_scale=0.25),
        dict(detection_scale=0.75),  # own INTER_AREA: bit-identical here
        dict(detection_scale=0.6),
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
    _check_against_reference(got, ref[0], ref[1])
    assert got.points_2d.shape[0] > 0


@pytest.mark.parametrize("band_rows", [1, 2, 3, 7, 96])
def test_parallel_tracer_matches_findcontours(band_rows):
    # The band-parallel contour tracer must return exactly the contours (points
    # and order) of sequential cv::findContours, even with bands of a single row
    # where nearly every contour crosses band boundaries.
    rng = np.random.default_rng(band_rows)
    total = 0
    for t in range(60):
        h, w = rng.integers(4, 160, size=2)
        if t % 3 == 0:
            img = (rng.random((h, w)) < rng.uniform(0.2, 0.8)).astype(np.uint8)
        else:  # blobs with holes, thin walls and nesting
            f = rng.random((h + 4, w + 4))
            f = (f[:-4, :-4] + f[2:-2, 2:-2] + f[4:, 4:] + f[2:-2, :-4] + f[:-4, 4:]) / 5
            img = (f > rng.uniform(0.4, 0.6)).astype(np.uint8)
        img = np.ascontiguousarray(img)
        for min_size in (1, 2, 30):
            same, n = _nf._tracer_matches_findcontours(img, min_size, band_rows)
            assert same, f"image {t} ({h}x{w}), min_size {min_size}"
            total += n
    assert total > 0


@pytest.mark.parametrize("threads", [1, 0])
@pytest.mark.parametrize("size", [(1920, 1080), (1281, 721), (97, 53)])
@pytest.mark.parametrize("f", [0.75, 0.6, 0.35, 0.3, 0.9, 0.123])
def test_area_downscale_matches_opencv(f, size, threads):
    # non-integer detection_scale uses a separable INTER_AREA; it may differ
    # from cv::resize by one grey level on a few pixels (float sum order)
    img = generate_scene(max(size[0], 200), max(size[1], 120), "clutter", seed=5)
    img = np.ascontiguousarray(img[: size[1], : size[0]])
    mx, n, same_size = _nf._area_downscale_diff(img, f, threads)
    assert same_size
    assert mx <= 1
    assert n <= 0.005 * img.size * f * f


@pytest.mark.parametrize("threads", [1, 0])
@pytest.mark.parametrize("k", [3, 7, 15, 17, 21, 31, 41, 61, 101, 255])
def test_threshold_matches_opencv(k, threads):
    # the box sums (any algorithm: they are exact integers) and the rounding
    # of the mean must reproduce cv::adaptiveThreshold bit for bit
    for w, h, seed in [(1920, 1080, 1), (1283, 721, 2), (37, 300, 3), (3840, 64, 4)]:
        img = generate_scene(max(w, 200), max(h, 120), "clutter", seed=seed)
        img = np.ascontiguousarray(img[:h, :w])
        for delta in (7.0, 0.0, -3.5):
            bad, bad_nbm = _nf._threshold_diff(img, k, delta, threads)
            assert bad == 0 and bad_nbm == 0, (w, h, delta, bad, bad_nbm)
