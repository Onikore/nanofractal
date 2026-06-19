"""Smoke test for nanofractal.bench CLI module."""

import io
import sys

from nanofractal import bench


def test_bench_smoke():
    """Quick smoke test: run bench with minimal frames and resolution."""
    # Capture stdout to avoid cluttering test output
    old_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        # Run with minimal settings for speed
        bench.main(["--frames", "2", "--resolution", "320x240", "--detector", "aruco"])
    finally:
        sys.stdout = old_stdout
    # If we get here, no exception was raised — test passes
