"""Smoke tests for nanofractal.bench CLI module."""

import io
import sys

import pytest

from nanofractal import bench

_SMALL = ["--frames", "2", "--resolution", "160x120"]


def _run(args):
    """Run bench.main with stdout suppressed; propagate exceptions."""
    old_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        bench.main(args)
    finally:
        sys.stdout = old_stdout


def test_bench_smoke_aruco():
    """Bench loop + report must complete without error for aruco detector."""
    _run([*_SMALL, "--detector", "aruco"])


def test_bench_smoke_fractal():
    """Bench loop + report must complete without error for fractal detector."""
    _run([*_SMALL, "--detector", "fractal"])


def test_bench_smoke_both():
    """Bench loop + report must complete without error for both detectors."""
    _run([*_SMALL, "--detector", "both"])


def test_bench_bad_resolution_exits(monkeypatch):
    """Invalid resolution string causes a clean SystemExit (not a crash)."""
    with pytest.raises(SystemExit):
        _run(["--frames", "2", "--resolution", "badres", "--detector", "aruco"])
