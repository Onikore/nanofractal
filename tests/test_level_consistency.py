"""Levels whose position does not fit the rest of the marker are dropped (R18)."""

import numpy as np
import pytest

import nanofractal as nf

CONFIG = "FRACTAL_3L_6"
SIDE = 300


def _marker_page(extra_level=None, scale=1.6):
    """Marker at (60, 60); optionally a bigger copy of one inner level elsewhere."""
    m = nf.generate_fractal(CONFIG, SIDE)
    page = np.full((SIDE + 120, 2 * SIDE + 300), 255, np.uint8)
    page[60 : 60 + SIDE, 60 : 60 + SIDE] = m
    if extra_level is not None:
        det = nf.FractalDetector(CONFIG)
        res = det.detect(page)
        c = res.corners[res.ids.tolist().index(extra_level)]
        x0, y0 = np.round(c.min(0)).astype(int)
        x1, y1 = np.round(c.max(0)).astype(int)
        lvl = page[y0 : y1 + 1, x0 : x1 + 1].copy()
        for mid, ci in zip(res.ids.tolist(), res.corners):
            if mid > extra_level:  # blank the nested levels (masked cells)
                a0, b0 = np.round(ci.min(0)).astype(int) - [x0, y0]
                a1, b1 = np.round(ci.max(0)).astype(int) - [x0, y0]
                lvl[b0 : b1 + 1, a0 : a1 + 1] = 0
        k = int(round(lvl.shape[0] * scale)) // lvl.shape[0] or 1
        big = np.kron(lvl, np.ones((k, k), np.uint8))
        ox, oy = SIDE + 160, 60 + (SIDE - big.shape[0]) // 2
        page[oy : oy + big.shape[0], ox : ox + big.shape[1]] = big
    return np.ascontiguousarray(page)


def test_all_levels_consistent():
    res = nf.FractalDetector(CONFIG).detect(_marker_page())
    assert res.ids.tolist() == [0, 1, 2]
    assert res.inconsistent_ids.tolist() == []


@pytest.mark.parametrize("inner", [False, True])
def test_false_level_elsewhere_dropped(inner):
    det = nf.FractalDetector(CONFIG, marker_size=0.2)
    page = _marker_page(extra_level=1, scale=2.0)
    res = det.detect(page, with_inner_points=inner)
    # the copy of level 1 (nested level blanked) is larger, so it wins the
    # duplicate check, but it does not sit where levels 0 and 2 put level 1
    assert res.inconsistent_ids.tolist() == [1]
    assert res.ids.tolist() == [0, 2]
    assert res.multiple_instances
    if inner:
        assert (
            det.estimate_pose(
                res, np.array([[800.0, 0, 400], [0, 800, 200], [0, 0, 1]]), np.zeros(5)
            )[2]
            < 1.0
        )


def test_batch_and_diagnose_report_it():
    det = nf.FractalDetector(CONFIG)
    page = _marker_page(extra_level=1, scale=2.0)
    (b,) = det.detect_batch([page])
    assert b.inconsistent_ids.tolist() == [1]
    d = det.diagnose(page)
    assert d.result.inconsistent_ids.tolist() == [1]
    assert [q.nearest_id for q in d.quads if q.reason == "inconsistent"] == [1]


@pytest.mark.parametrize("config", ["FRACTAL_3L_6", "FRACTAL_5L_6"])
def test_points_level(config):
    # R19: the level of every inner point; a point sitting on a level's outer
    # corner is labelled with that level
    from nanofractal.bench import generate_scene

    det = nf.FractalDetector(config, marker_size=0.2)
    res = det.detect(
        generate_scene(1280, 720, "perspective", config=config), with_inner_points=True
    )
    assert res.points_level is not None
    assert res.points_level.dtype == np.int32
    assert res.points_level.shape == (res.points_2d.shape[0],)
    assert set(res.points_level.tolist()) <= set(res.ids.tolist())
    checked = 0
    for mid, corners in res:
        assert (res.points_level == mid).sum() >= 1
        for c in corners:
            d = np.linalg.norm(res.points_2d - c, axis=1)
            if d.min() < 0.5:
                assert res.points_level[d.argmin()] == mid
                checked += 1
    assert checked >= 3 * len(res)
    assert nf.FractalDetector(config).detect(generate_scene(640, 480, "clean")).points_level is None
