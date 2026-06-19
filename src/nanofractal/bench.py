"""Self-contained benchmark CLI for nanofractal detectors.

Run as: python -m nanofractal.bench [options]
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time

import numpy as np

import nanofractal as nf


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse 'WxH' string to (width, height)."""
    parts = res_str.split("x")
    if len(parts) != 2:
        raise ValueError(f"Resolution must be WxH format, got {res_str!r}")
    try:
        w, h = int(parts[0]), int(parts[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"Width and height must be positive, got {w}x{h}")
        return w, h
    except ValueError as e:
        raise ValueError(f"Invalid resolution format {res_str!r}: {e}")


def generate_test_frame(width: int, height: int, detector_type: str) -> np.ndarray:
    """Generate a grayscale test frame with a marker near the top-left.

    Parameters
    ----------
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    detector_type : str
        One of 'aruco', 'fractal', or 'both'.

    Returns
    -------
    np.ndarray
        Grayscale uint8 array of shape (height, width), C-contiguous.
    """
    # Create white canvas
    frame = np.full((height, width), 255, dtype=np.uint8)

    # Determine marker size (at most 1/3 of smallest frame dimension)
    max_marker_size = min(height, width) // 3
    # Clamp to ensure it fits even in very small frames
    max_marker_size = max(8, max_marker_size)  # Minimum 8px for visibility

    # For 'both', we'll include both markers at different positions
    # For single detector, use the appropriate marker type
    marker = None
    if detector_type == "aruco":
        marker_size = max_marker_size
        try:
            marker = nf.generate_aruco(0, size_px=marker_size, dictionary=nf.Dict.DICT_4X4_50)
        except Exception:
            # If marker generation fails, skip
            pass
    elif detector_type == "fractal":
        marker_size = max_marker_size
        try:
            marker = nf.generate_fractal("FRACTAL_5L_6", size_px=marker_size)
        except Exception:
            # If marker generation fails, skip
            pass
    elif detector_type == "both":
        # Include both markers at different locations
        marker_size = max_marker_size
        try:
            marker_aruco = nf.generate_aruco(0, size_px=marker_size, dictionary=nf.Dict.DICT_4X4_50)
        except Exception:
            marker_aruco = None

        try:
            marker_fractal = nf.generate_fractal("FRACTAL_5L_6", size_px=marker_size)
        except Exception:
            marker_fractal = None

        # Paste ArUco at top-left, Fractal at top-right
        margin = max(10, min(height, width) // 100)

        if marker_aruco is not None:
            marker_h, marker_w = marker_aruco.shape
            y_start = margin
            x_start = margin
            y_end = min(y_start + marker_h, height)
            x_end = min(x_start + marker_w, width)
            if y_end > y_start and x_end > x_start:
                marker_y_slice = slice(0, y_end - y_start)
                marker_x_slice = slice(0, x_end - x_start)
                frame[y_start:y_end, x_start:x_end] = marker_aruco[marker_y_slice, marker_x_slice]

        if marker_fractal is not None:
            marker_h, marker_w = marker_fractal.shape
            # Paste at top-right (with right margin)
            y_start = margin
            x_start = max(width - marker_w - margin, 0)
            y_end = min(y_start + marker_h, height)
            x_end = min(x_start + marker_w, width)
            if y_end > y_start and x_end > x_start:
                marker_y_slice = slice(0, y_end - y_start)
                # Handle marker slicing from the right
                marker_x_start = max(0, marker_w - (x_end - x_start))
                marker_x_slice = slice(marker_x_start, marker_w)
                frame[y_start:y_end, x_start:x_end] = marker_fractal[marker_y_slice, marker_x_slice]

        # Return early for 'both' since we already pasted both markers
        return np.ascontiguousarray(frame)

    # Paste marker near top-left with margin
    if marker is not None:
        marker_h, marker_w = marker.shape
        # Add a small margin (10 pixels or 1% of frame, whichever is larger)
        margin = max(10, min(height, width) // 100)

        # Calculate paste position with margin
        y_start = margin
        x_start = margin

        # Calculate paste end positions, clamped to frame boundaries
        y_end = min(y_start + marker_h, height)
        x_end = min(x_start + marker_w, width)

        # Only paste if there's space
        if y_end > y_start and x_end > x_start:
            # Adjust marker region to fit within frame boundaries
            marker_y_slice = slice(0, y_end - y_start)
            marker_x_slice = slice(0, x_end - x_start)
            frame[y_start:y_end, x_start:x_end] = marker[marker_y_slice, marker_x_slice]

    return np.ascontiguousarray(frame)


def benchmark_detector(
    name: str, detector, frame: np.ndarray, num_frames: int, num_threads: int = 0
) -> dict:
    """Benchmark a single detector.

    Parameters
    ----------
    name : str
        Detector name for reporting.
    detector : ArucoDetector or FractalDetector
        Detector instance.
    frame : np.ndarray
        Grayscale uint8 test frame.
    num_frames : int
        Number of detection calls to time.
    num_threads : int
        Number of threads (for batch-mode testing; default 0 = single-frame).

    Returns
    -------
    dict
        Dictionary with keys: 'name', 'frames', 'mean_ms', 'median_ms', 'p95_ms',
        'fps', 'detected' (bool).
    """
    # Warm up
    for _ in range(3):
        detector.detect(frame)

    # Measure
    latencies = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        result = detector.detect(frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # Convert to ms

    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    p95_ms = np.percentile(latencies, 95)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    detected = len(result) > 0

    return {
        "name": name,
        "frames": num_frames,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "fps": fps,
        "detected": detected,
    }


def print_report(width: int, height: int, results: list[dict]) -> None:
    """Print a formatted benchmark report.

    Parameters
    ----------
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    results : list[dict]
        List of detector benchmark results.
    """
    print("\n" + "=" * 70)
    print("NANOFRACTAL BENCHMARK REPORT")
    print("=" * 70)
    print(f"nanofractal version: {nf.__version__}")
    print(f"OpenCV version:      {nf._opencv_version()}")
    print(f"CPU arch:            {platform.machine()}")

    proc = platform.processor()
    if proc:
        print(f"Processor:           {proc}")

    print(f"Python version:      {platform.python_version()}")
    print(f"Resolution:          {width}x{height}")
    print("-" * 70)

    for res in results:
        print(f"\n{res['name']}:")
        print(f"  Frames:    {res['frames']}")
        print(f"  Mean:      {res['mean_ms']:.2f} ms")
        print(f"  Median:    {res['median_ms']:.2f} ms")
        print(f"  P95:       {res['p95_ms']:.2f} ms")
        print(f"  FPS:       {res['fps']:.1f}")
        print(f"  Detected:  {res['detected']}")

    print("\n" + "=" * 70)


def main(argv: list[str] | None = None) -> None:
    """Main entry point.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. If None, sys.argv[1:] is used.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark nanofractal detectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
        help="Frame resolution as WxH (default: 1280x720)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["aruco", "fractal", "both"],
        default="both",
        help="Which detector(s) to benchmark",
    )
    parser.add_argument(
        "--frames", type=int, default=200, help="Number of frames to benchmark (default: 200)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of threads (for batch-mode; 0 = single-frame, default: 0)",
    )

    args = parser.parse_args(argv)

    # Parse resolution
    try:
        width, height = parse_resolution(args.resolution)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate frames
    if args.frames <= 0:
        print(f"Error: --frames must be > 0, got {args.frames}", file=sys.stderr)
        sys.exit(1)

    # Generate test frame
    print(f"Generating test frame ({width}x{height})...")
    frame = generate_test_frame(width, height, args.detector)

    results = []

    # Benchmark ArUco if requested
    if args.detector in ("aruco", "both"):
        print("Benchmarking ArUco detector...")
        aruco_det = nf.ArucoDetector(nf.Dict.DICT_4X4_50, max_attempts=1)
        results.append(benchmark_detector("ArUco", aruco_det, frame, args.frames, args.threads))

    # Benchmark Fractal if requested
    if args.detector in ("fractal", "both"):
        print("Benchmarking Fractal detector...")
        fractal_det = nf.FractalDetector("FRACTAL_5L_6")
        results.append(benchmark_detector("Fractal", fractal_det, frame, args.frames, args.threads))

    # Print report
    print_report(width, height, results)


if __name__ == "__main__":
    main()
