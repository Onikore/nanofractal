"""benchmarks/robustness.py — fractal detection quality vs synthetic degradations.

For every degradation level it reports which nested levels were found, how
many inner-corner correspondences were matched, and the RMS reprojection
error of the resulting pose. Deterministic (fixed seeds), so two runs — e.g.
before and after a performance change — can be diffed via ``--json``.

Usage:
    python benchmarks/robustness.py
    python benchmarks/robustness.py --json results.json
    python benchmarks/robustness.py --real-dir benchmarks/real_images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import nanofractal as nf
from nanofractal.bench import _blur, _warp_onto

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

_CONFIG = "FRACTAL_5L_6"
_CANVAS = 1280
_MARKER_PX = 780
_FOCAL = 900.0


def _make_detector() -> nf.FractalDetector:
    return nf.FractalDetector(_CONFIG, marker_size=1.0)


def _scene(marker_px: int = _MARKER_PX, quad_jitter: np.ndarray | None = None) -> np.ndarray:
    """Nested marker (with quiet zone) warped onto a white canvas."""
    quiet = marker_px // 12
    m = np.full((marker_px + 2 * quiet, marker_px + 2 * quiet), 255, np.uint8)
    m[quiet : quiet + marker_px, quiet : quiet + marker_px] = nf.generate_fractal(
        _CONFIG, marker_px
    )
    canvas = np.full((_CANVAS, _CANVAS), 255.0)
    half = m.shape[0] / 2
    c = _CANVAS / 2
    quad = np.array([[c - half, c - half], [c + half, c - half], [c + half, c + half],
                     [c - half, c + half]])
    if quad_jitter is not None:
        quad = quad + quad_jitter
    _warp_onto(canvas, m, quad)
    return canvas


def _finish(img: np.ndarray, blur: float = 0.0, noise: float = 0.0) -> np.ndarray:
    img = _blur(img, blur)
    if noise > 0:
        img = img + np.random.default_rng(42).normal(0, noise, img.shape)
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


def _evaluate(img: np.ndarray) -> dict[str, Any]:
    det = _make_detector()
    res = det.detect(img, with_inner_points=True)
    K = np.array([[_FOCAL, 0, _CANVAS / 2], [0, _FOCAL, _CANVAS / 2], [0, 0, 1]], np.float64)
    pose = det.estimate_pose(res, K, np.zeros(5))
    return {
        "levels": sorted(res.ids.tolist()),
        "inner_pts": int(res.points_2d.shape[0]),
        "reproj_px": round(float(pose[2]), 3) if pose is not None else None,
    }


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")
    if not rows:
        print("  (no data)")
        return
    keys = list(rows[0].keys())
    widths = [max(len(k), max(len(str(r[k])) for r in rows)) + 2 for k in keys]
    header = "".join(k.ljust(w) for k, w in zip(keys, widths))
    print("  " + header + "\n  " + "─" * len(header))
    for row in rows:
        print("  " + "".join(str(row[k]).ljust(w) for k, w in zip(keys, widths)))


def sweep_blur() -> list[dict]:
    rows = [{"sigma": s, **_evaluate(_finish(_scene(), blur=s))} for s in [0, 1, 2, 3, 5]]
    _print_table("Blur sweep (Gaussian sigma, px)", rows)
    return rows


def sweep_noise() -> list[dict]:
    rows = [{"noise_sigma": s, **_evaluate(_finish(_scene(), noise=s))} for s in [0, 5, 10, 20, 40]]
    _print_table("Noise sweep (additive Gaussian sigma, 0-255 range)", rows)
    return rows


def _tilted_jitter(deg: float) -> np.ndarray:
    """Corner offsets of the marker square tilted *deg* about the x axis, as
    seen by the pinhole camera used in _evaluate (a physically valid view)."""
    side = _MARKER_PX * 7 / 6  # marker + quiet zone, px when frontal
    z = _FOCAL  # frontal: 1 world unit == 1 px at depth f
    h = side / 2
    a = np.deg2rad(deg)
    pts = []
    for x, y in [(-h, -h), (h, -h), (h, h), (-h, h)]:
        yr, zr = y * np.cos(a), z + y * np.sin(a)
        pts.append([_FOCAL * x / zr, _FOCAL * yr / zr])
    base = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    return np.array(pts) - base


def sweep_perspective() -> list[dict]:
    rows = []
    for deg in [0, 15, 30, 45, 60]:
        rows.append({"tilt_deg": deg, **_evaluate(_finish(_scene(quad_jitter=_tilted_jitter(deg))))})
    _print_table("Perspective sweep (marker tilted about the x axis)", rows)
    return rows


def sweep_rotation() -> list[dict]:
    rows = []
    for angle in [0, 5, 15, 30, 45]:
        side = _MARKER_PX * 7 / 6
        a = np.deg2rad(angle)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * side / 2
        jitter = base @ rot.T - base
        rows.append({"angle_deg": angle, **_evaluate(_finish(_scene(quad_jitter=jitter)))})
    _print_table("Rotation sweep (in-plane degrees)", rows)
    return rows


def sweep_scale() -> list[dict]:
    rows = []
    for px in [1100, 780, 400, 200, 120]:
        rows.append({"marker_px": px, **_evaluate(_finish(_scene(marker_px=px), blur=0.7))})
    _print_table(f"Scale sweep (marker side in px, canvas {_CANVAS}x{_CANVAS})", rows)
    return rows


def run_real_images(real_dir: str | Path) -> list[dict]:
    from benchmarks._real import iter_real_images  # noqa: PLC0415

    rows = []
    total = correct_count = 0
    try:
        for img, config, expected_ids in iter_real_images(real_dir):
            total += 1
            detected = nf.FractalDetector(config).detect(img).ids.tolist()
            correct = all(eid in detected for eid in expected_ids)
            correct_count += int(correct)
            rows.append(
                {"expected": str(expected_ids), "detected": str(detected),
                 "correct": "YES" if correct else "NO"}
            )
    except FileNotFoundError as exc:
        print(f"\n[real-images] {exc}")
        return []
    pct = 100 * correct_count / max(total, 1)
    _print_table(f"Real images — {correct_count}/{total} correct ({pct:.0f}%)", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure fractal detection robustness under synthetic degradations."
    )
    parser.add_argument("--real-dir", metavar="DIR", default=None,
                        help="Run real-image evaluation from DIR (needs DIR/manifest.json).")
    parser.add_argument("--json", metavar="PATH", help="Write JSON results to PATH")
    args = parser.parse_args()

    print(f"\nnanofractal {nf.__version__}  OpenCV {nf._opencv_version()}  config {_CONFIG}")
    results: dict[str, Any] = {
        "blur": sweep_blur(),
        "noise": sweep_noise(),
        "perspective": sweep_perspective(),
        "rotation": sweep_rotation(),
        "scale": sweep_scale(),
    }
    if args.real_dir:
        if not _HAS_CV2:
            print("Skipping real images: opencv-python not installed")
        else:
            results["real_images"] = run_real_images(args.real_dir)
    print()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"JSON results written to {args.json}")


if __name__ == "__main__":
    main()
