"""benchmarks/reading.py — far markers, shadow edges and marker-like negatives.

Synthetic frames for the bit reader (ROADMAP R16, R9): a FRACTAL_4L_6 marker
at a few pixels per cell, under a hard shadow, partly occluded, and frames full of
marker-like objects (random codes with a black border, markers of other
configurations) that must not be detected. Deterministic (fixed seeds).

Usage:
    python benchmarks/reading.py
    python benchmarks/reading.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import nanofractal as nf
from nanofractal.bench import _background, _blur, _warp_onto

CONFIG = "FRACTAL_4L_6"
CELLS = 15  # outer grid of FRACTAL_4L_6, border included
W, H = 1280, 720
SS = 6  # supersampling of the marker render


def _marker(size: int = 1200) -> np.ndarray:
    quiet = size // 12
    m = np.full((size + 2 * quiet,) * 2, 255.0)
    m[quiet:-quiet, quiet:-quiet] = nf.generate_fractal(CONFIG, size)
    return m


def render(rng, side, shadow=None, tilt=0.15, occlude=0.0):
    """Marker with a `side` px black square, random pose, on clutter."""
    img = _background(W, H, rng)
    mm = _marker()
    half = side / 2 * mm.shape[0] / 1200
    a = rng.uniform(0, 2 * np.pi)
    c = np.array([W / 2, H / 2]) + rng.uniform(-20, 20, 2)
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64) * half
    base = base * (1 + rng.uniform(-tilt, tilt, (4, 1)))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    quad = base @ rot.T + c
    x0, y0 = np.floor(quad.min(0)).astype(int) - 2
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 2
    sub = np.zeros(((y1 - y0) * SS, (x1 - x0) * SS))
    cov = np.zeros_like(sub)
    _warp_onto(sub, mm, (quad - [x0, y0]) * SS)
    _warp_onto(cov, np.ones_like(mm), (quad - [x0, y0]) * SS)
    sub = sub.reshape(y1 - y0, SS, x1 - x0, SS).mean((1, 3))
    cov = cov.reshape(y1 - y0, SS, x1 - x0, SS).mean((1, 3))
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - cov) + sub
    if occlude > 0:  # a dark or bright disc over `occlude` of the marker area
        rad = side * np.sqrt(occlude / np.pi)
        oc = c + rng.uniform(-0.5, 0.5, 2) * side
        yy, xx = np.mgrid[0:H, 0:W]
        img[(xx - oc[0]) ** 2 + (yy - oc[1]) ** 2 < rad * rad] = rng.choice([20.0, 235.0])
    if shadow is not None:  # half-plane through the marker, soft 2 px edge
        b = rng.uniform(0, 2 * np.pi)
        off = rng.uniform(-0.25, 0.25) * side
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        dist = (xx - c[0]) * np.cos(b) + (yy - c[1]) * np.sin(b) - off
        img = img * (shadow + (1 - shadow) * np.clip(dist / 2.0 + 0.5, 0, 1))
    img = _blur(img, 0.6) + rng.normal(0, 3, img.shape)
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


def negative(rng):
    """Clutter plus 12 marker-like objects, none of them a FRACTAL_4L_6."""
    img = _background(W, H, rng)
    for _ in range(12):
        if rng.integers(0, 2) == 0:
            n = int(rng.integers(6, 18))
            g = (rng.random((n, n)) > 0.5) * 255.0
            g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 0
            src = np.kron(g, np.ones((40, 40)))
        else:
            cfg = ["FRACTAL_2L_6", "FRACTAL_3L_6", "FRACTAL_5L_6"][rng.integers(0, 3)]
            src = nf.generate_fractal(cfg, 600).astype(np.float64)
            n = 15
        pad = np.full((src.shape[0] + 160, src.shape[1] + 160), 255.0)
        pad[80:-80, 80:-80] = src
        side = rng.uniform(2.0, 8.0) * n
        c = rng.uniform([80, 80], [W - 80, H - 80])
        a = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        half = side / 2 * pad.shape[0] / src.shape[0]
        base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
        quad = (base * (1 + rng.uniform(-0.15, 0.15, (4, 1)))) @ rot.T + c
        _warp_onto(img, pad, quad)
    img = _blur(img, rng.uniform(0.4, 1.2)) + rng.normal(0, 3, img.shape)
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, default=60, help="frames per scenario")
    ap.add_argument("--negatives", type=int, default=300)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()
    det = nf.FractalDetector(CONFIG)
    out: dict[str, float] = {}
    print(f"nanofractal {nf.__version__}  {CONFIG}  {W}x{H}  outer level found:")
    for ppc in [2.2, 2.6, 3.0, 3.5, 4.4]:
        rng = np.random.default_rng(int(ppc * 10))
        frames = [render(rng, ppc * CELLS) for _ in range(args.frames)]
        out[f"far_{ppc}"] = np.mean([0 in det.detect(f).ids.tolist() for f in frames])
        print(f"  far {ppc} px/cell   {out[f'far_{ppc}']:6.0%}")
    for dark in [0.6, 0.45, 0.3]:
        rng = np.random.default_rng(int(dark * 100))
        frames = [render(rng, 10 * CELLS, shadow=dark) for _ in range(args.frames)]
        out[f"shadow_{dark}"] = np.mean([0 in det.detect(f).ids.tolist() for f in frames])
        print(f"  shadow x{dark}       {out[f'shadow_{dark}']:6.0%}")
    for frac in [0.1, 0.25, 0.4]:
        rng = np.random.default_rng(int(frac * 100) + 1)
        frames = [render(rng, 10 * CELLS, occlude=frac) for _ in range(args.frames)]
        found = [det.detect(f).ids.tolist() for f in frames]
        out[f"occlusion_{frac}_outer"] = np.mean([0 in ids for ids in found])
        out[f"occlusion_{frac}_any"] = np.mean([len(ids) > 0 for ids in found])
        print(
            f"  occluded {frac:.0%}        {out[f'occlusion_{frac}_outer']:6.0%}"
            f"   (any level: {out[f'occlusion_{frac}_any']:.0%})"
        )
    rng = np.random.default_rng(7)
    neg = [negative(rng) for _ in range(args.negatives)]
    t0 = time.perf_counter()
    out["negative_frames_with_detection"] = sum(len(det.detect(f)) > 0 for f in neg)
    ms = (time.perf_counter() - t0) / len(neg) * 1e3
    print(
        f"  negatives with a detection: {out['negative_frames_with_detection']}/{len(neg)}"
        f"  ({ms:.2f} ms/frame)"
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump({k: float(v) for k, v in out.items()}, f, indent=1)


if __name__ == "__main__":
    main()
