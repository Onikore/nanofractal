"""Smoke tests for nanofractal.bench CLI module."""

import io
import sys

import numpy as np
import pytest

from nanofractal import bench

_SMALL = ["--frames", "2", "--resolution", "320x240"]


def _run(args):
    """Run bench.main with stdout suppressed; propagate exceptions."""
    old_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        bench.main(args)
    finally:
        sys.stdout = old_stdout


@pytest.mark.parametrize("scene", bench.SCENES)
def test_bench_smoke_scenes(scene):
    _run([*_SMALL, "--scene", scene])


@pytest.mark.parametrize("mode", bench.MODES)
def test_bench_smoke_modes(mode):
    _run([*_SMALL, "--mode", mode])


def test_bench_bad_resolution_exits():
    """Invalid resolution string causes a clean SystemExit (not a crash)."""
    with pytest.raises(SystemExit):
        _run(["--frames", "2", "--resolution", "badres"])


def test_scene_is_deterministic():
    a = bench.generate_scene(320, 240, "perspective", seed=3)
    b = bench.generate_scene(320, 240, "perspective", seed=3)
    assert a.dtype == np.uint8 and a.shape == (240, 320) and a.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("scene", ["clean", "perspective"])
def test_scene_marker_is_detectable(scene):
    frame = bench.generate_scene(1280, 720, scene)
    res = bench.make_detector("detect").detect(frame)
    assert 0 in res.ids.tolist()  # external level of FRACTAL_5L_6
