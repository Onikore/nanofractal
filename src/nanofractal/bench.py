"""Self-contained benchmark CLI and scene generator for the fractal detector.

Run as: python -m nanofractal.bench [options]

Scenes are synthesised with numpy only (seeded, deterministic), so the same
command produces identical frames on every machine:

``clean``
    The full nested marker, frontal, on a white background.
``perspective``
    The nested marker under a perspective warp over a cluttered, unevenly lit
    background, with blur and sensor noise. The realistic case.
``clutter``
    The same cluttered background with no marker. Exercises the threshold and
    contour stages, which dominate the cost when nothing is found.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time

import numpy as np

import nanofractal as nf

SCENES = ("clean", "perspective", "clutter")
MODES = ("detect", "inner", "scale05")


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse 'WxH' string to (width, height)."""
    parts = res_str.split("x")
    if len(parts) != 2:
        raise ValueError(f"Resolution must be WxH format, got {res_str!r}")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise ValueError(f"Invalid resolution format {res_str!r}: {e}") from e
    if w <= 0 or h <= 0:
        raise ValueError(f"Width and height must be positive, got {w}x{h}")
    return w, h


# ---------------------------------------------------------------------------
# numpy-only image synthesis
# ---------------------------------------------------------------------------


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping 4 src points onto 4 dst points (DLT, exact)."""
    a = []
    b = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    h = np.linalg.solve(np.asarray(a, np.float64), np.asarray(b, np.float64))
    return np.append(h, 1.0).reshape(3, 3)


def _warp_onto(canvas: np.ndarray, src: np.ndarray, quad: np.ndarray) -> None:
    """Bilinearly warp *src* so its corners land on *quad* (TL,TR,BR,BL), in place."""
    sh, sw = src.shape
    hinv = np.linalg.inv(
        _homography(np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float64), quad)
    )
    x0, y0 = np.floor(quad.min(0)).astype(int)
    x1, y1 = np.ceil(quad.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, canvas.shape[1]), min(y1, canvas.shape[0])
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    xs += 0.5
    ys += 0.5
    w = hinv[2, 0] * xs + hinv[2, 1] * ys + hinv[2, 2]
    u = (hinv[0, 0] * xs + hinv[0, 1] * ys + hinv[0, 2]) / w - 0.5
    v = (hinv[1, 0] * xs + hinv[1, 1] * ys + hinv[1, 2]) / w - 0.5
    inside = (u >= -0.5) & (u <= sw - 0.5) & (v >= -0.5) & (v <= sh - 0.5)
    u = np.clip(u, 0, sw - 1)
    v = np.clip(v, 0, sh - 1)
    iu = np.minimum(u.astype(np.intp), sw - 2)
    iv = np.minimum(v.astype(np.intp), sh - 2)
    fu, fv = u - iu, v - iv
    s = src.astype(np.float64)
    val = (
        s[iv, iu] * (1 - fu) * (1 - fv)
        + s[iv, iu + 1] * fu * (1 - fv)
        + s[iv + 1, iu] * (1 - fu) * fv
        + s[iv + 1, iu + 1] * fu * fv
    )
    region = canvas[y0:y1, x0:x1]
    region[inside] = val[inside]


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (float64 in/out)."""
    if sigma <= 0:
        return img
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(img, r, mode="edge")
    tmp = sum(k[i] * pad[:, i : i + img.shape[1]] for i in range(2 * r + 1))
    return sum(k[i] * tmp[i : i + img.shape[0], :] for i in range(2 * r + 1))


def _background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Unevenly lit background with rectangles, discs and thin lines (float64)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    gx, gy = rng.uniform(-60, 60, 2)
    img = 170 + gx * (xx / width - 0.5) + gy * (yy / height - 0.5)
    scale = np.sqrt(width * height) / 1000.0
    n_shapes = int(120 * scale * scale) + 20
    for _ in range(n_shapes):
        kind = rng.integers(0, 3)
        val = rng.uniform(20, 240)
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        size = rng.uniform(8, 90) * scale
        x0, x1 = int(max(cx - size, 0)), int(min(cx + size, width))
        y0, y1 = int(max(cy - size, 0)), int(min(cy + size, height))
        if x1 <= x0 or y1 <= y0:
            continue
        if kind == 0:  # rectangle
            img[y0:y1, x0:x1] = val
        elif kind == 1:  # disc
            sub_y, sub_x = yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]
            m = (sub_x - cx) ** 2 + (sub_y - cy) ** 2 < size * size
            img[y0:y1, x0:x1][m] = val
        else:  # thin line
            t = max(1, int(2 * scale))
            if rng.integers(0, 2):
                img[int(cy) : int(cy) + t, x0:x1] = val
            else:
                img[y0:y1, int(cx) : int(cx) + t] = val
    return img


def generate_scene(
    width: int, height: int, scene: str = "perspective", config: str = "FRACTAL_5L_6", seed: int = 0
) -> np.ndarray:
    """Synthesise a grayscale uint8 benchmark frame (see module docstring)."""
    if scene not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}, got {scene!r}")
    rng = np.random.default_rng(seed)
    side = int(min(width, height) * 0.6)
    if scene == "clean":
        img = np.full((height, width), 255.0)
        m = nf.generate_fractal(config, size_px=side).astype(np.float64)
        y0, x0 = (height - side) // 2, (width - side) // 2
        img[y0 : y0 + side, x0 : x0 + side] = m
    else:
        img = _background(width, height, rng)
        if scene == "perspective":
            quiet = side // 12
            m = np.full((side + 2 * quiet, side + 2 * quiet), 255, np.uint8)
            m[quiet : quiet + side, quiet : quiet + side] = nf.generate_fractal(config, side)
            cx, cy = width * 0.5, height * 0.5
            half = m.shape[0] / 2
            quad = np.array(
                [[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half],
                 [cx - half, cy + half]]
            )
            quad += rng.uniform(-0.12, 0.12, (4, 2)) * m.shape[0]
            _warp_onto(img, m, quad)
        img = _blur(img, 0.8 * max(1.0, min(width, height) / 720))
        img += rng.normal(0, 3.0, img.shape)
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def make_detector(mode: str, config: str = "FRACTAL_5L_6") -> nf.FractalDetector:
    params = nf.DetectorParams()
    if mode == "scale05":
        params.detection_scale = 0.5
    return nf.FractalDetector(config, marker_size=1.0, params=params)


def run_once(det: nf.FractalDetector, frame: np.ndarray, mode: str) -> nf.DetectionResult:
    return det.detect(frame, with_inner_points=(mode == "inner"))


def benchmark_detector(
    name: str, detector, frame: np.ndarray, num_frames: int, mode: str = "detect"
) -> dict:
    """Time *num_frames* single-frame detections; returns summary statistics."""
    for _ in range(3):  # warm up
        result = run_once(detector, frame, mode)
    latencies = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        result = run_once(detector, frame, mode)
        latencies.append((time.perf_counter() - t0) * 1000)
    mean_ms = statistics.mean(latencies)
    return {
        "name": name,
        "frames": num_frames,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "p95_ms": float(np.percentile(latencies, 95)),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "detected": len(result),
        "ids": result.ids.tolist(),
    }


def print_report(width: int, height: int, results: list[dict]) -> None:
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
        print(f"  Mean:      {res['mean_ms']:.3f} ms")
        print(f"  Median:    {res['median_ms']:.3f} ms")
        print(f"  Min:       {res['min_ms']:.3f} ms")
        print(f"  P95:       {res['p95_ms']:.3f} ms")
        print(f"  FPS:       {res['fps']:.1f}")
        print(f"  Detected:  {res['detected']} {res['ids']}")
    print("\n" + "=" * 70)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the nanofractal fractal detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--resolution", type=str, default="1280x720", help="frame size WxH")
    parser.add_argument("--scene", choices=SCENES, default="perspective")
    parser.add_argument("--mode", choices=MODES, default="detect",
                        help="detect | inner (with_inner_points) | scale05 (detection_scale=0.5)")
    parser.add_argument("--config", default="FRACTAL_5L_6")
    parser.add_argument("--frames", type=int, default=200, help="timed iterations")
    args = parser.parse_args(argv)

    try:
        width, height = parse_resolution(args.resolution)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if args.frames <= 0:
        print(f"Error: --frames must be > 0, got {args.frames}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating {args.scene} frame ({width}x{height})...")
    frame = generate_scene(width, height, args.scene, args.config)
    det = make_detector(args.mode, args.config)
    name = f"Fractal {args.config} [{args.scene}, {args.mode}]"
    print_report(width, height, [benchmark_detector(name, det, frame, args.frames, args.mode)])


if __name__ == "__main__":
    main()
