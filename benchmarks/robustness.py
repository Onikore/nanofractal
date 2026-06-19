"""benchmarks/robustness.py — detection rate + pose error vs synthetic degradations.

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

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DICT = nf.Dict.DICT_4X4_50
_MARKER_ID = 0
_CANVAS_SIZE = 640  # pixels — square canvas for the base scene
_BASE_MARKER_PX = 200
_ATTEMPTS = 10  # max_attempts for the detector (helps degraded images)


def _make_detector() -> nf.ArucoDetector:
    return nf.ArucoDetector(_DICT, max_attempts=_ATTEMPTS)


# ---------------------------------------------------------------------------
# Base scene
# ---------------------------------------------------------------------------


def _base_scene(marker_px: int = _BASE_MARKER_PX, canvas_px: int = _CANVAS_SIZE) -> np.ndarray:
    """White canvas with marker id 0 centred."""
    marker = nf.generate_aruco(_MARKER_ID, size_px=marker_px, dictionary=_DICT, border_bits=1)
    canvas = np.full((canvas_px, canvas_px), 255, dtype=np.uint8)
    offset = (canvas_px - marker_px) // 2
    canvas[offset : offset + marker_px, offset : offset + marker_px] = marker
    return canvas


# ---------------------------------------------------------------------------
# Degradation helpers  (all require cv2)
# ---------------------------------------------------------------------------


def _require_cv2(name: str) -> None:
    if not _HAS_CV2:
        print(f"Skipping {name}: opencv-python not installed (pip install nanofractal[bench])")


def _apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return img.copy()
    ksize = int(sigma * 6) | 1  # odd kernel
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _apply_rotation(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255)


def _apply_perspective(img: np.ndarray, tilt_frac: float) -> np.ndarray:
    """Horizontal tilt: shift the top edge inward by *tilt_frac* of image width."""
    h, w = img.shape[:2]
    shift = int(tilt_frac * w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[shift, 0], [w - shift, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=255)


def _apply_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return img.copy()
    noise = np.random.default_rng(42).normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------


def _detect(img: np.ndarray, detector: nf.ArucoDetector | None = None) -> bool:
    if detector is None:
        detector = _make_detector()
    result = detector.detect(img)
    return _MARKER_ID in result.ids


def _detect_and_corners(img: np.ndarray) -> tuple[bool, np.ndarray | None]:
    detector = _make_detector()
    result = detector.detect(img)
    if _MARKER_ID not in result.ids:
        return False, None
    idx = list(result.ids).index(_MARKER_ID)
    return True, result.corners[idx : idx + 1]


# ---------------------------------------------------------------------------
# Pose error helper
# ---------------------------------------------------------------------------

_FOCAL = 500.0  # synthetic focal length (pixels)


def _synthetic_K(size: int = _CANVAS_SIZE) -> np.ndarray:
    return np.array([[_FOCAL, 0, size / 2], [0, _FOCAL, size / 2], [0, 0, 1]], dtype=np.float64)


def _reproj_error(corners: np.ndarray) -> float:
    """Estimate pose for corners and return reprojection error (pixels)."""
    K = _synthetic_K()
    D = np.zeros(4, dtype=np.float64)
    detector = _make_detector()
    _, _, err = detector.estimate_pose(
        corners, K, D, marker_size=0.1, return_reproj=True, fisheye=False
    )
    return float(err[0])


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    if not rows:
        print("  (no data)")
        return
    col_keys = list(rows[0].keys())
    col_w = [max(len(k), max(len(str(r[k])) for r in rows)) + 2 for k in col_keys]
    header = "".join(k.ljust(w) for k, w in zip(col_keys, col_w))
    print("  " + header)
    print("  " + "─" * len(header))
    for row in rows:
        print("  " + "".join(str(row[k]).ljust(w) for k, w in zip(col_keys, col_w)))


# ---------------------------------------------------------------------------
# Sweep functions
# ---------------------------------------------------------------------------


def sweep_blur() -> list[dict]:
    if not _HAS_CV2:
        _require_cv2("blur sweep")
        return []
    rows = []
    base = _base_scene()
    det = _make_detector()
    for sigma in [0, 1, 2, 3, 5]:
        img = _apply_blur(base, sigma)
        found = _detect(img, det)
        rows.append({"sigma": sigma, "detected": "YES" if found else "NO"})
    _print_table("Blur sweep (GaussianBlur sigma)", rows)
    return rows


def sweep_rotation() -> list[dict]:
    if not _HAS_CV2:
        _require_cv2("rotation sweep")
        return []
    rows = []
    base = _base_scene()
    for angle in [0, 5, 15, 30, 45]:
        img = _apply_rotation(base, angle)
        found, corners = _detect_and_corners(img)
        if found and corners is not None:
            err = _reproj_error(corners)
            err_str = f"{err:.2f}"
        else:
            err_str = "—"
        rows.append(
            {
                "angle_deg": angle,
                "detected": "YES" if found else "NO",
                "reproj_err_px": err_str,
            }
        )
    _print_table("Rotation sweep (in-plane degrees)", rows)
    return rows


def sweep_perspective() -> list[dict]:
    if not _HAS_CV2:
        _require_cv2("perspective sweep")
        return []
    rows = []
    base = _base_scene()
    for tilt in [0.0, 0.05, 0.10, 0.20, 0.30]:
        img = _apply_perspective(base, tilt)
        found, corners = _detect_and_corners(img)
        if found and corners is not None:
            err = _reproj_error(corners)
            err_str = f"{err:.2f}"
        else:
            err_str = "—"
        rows.append(
            {
                "tilt_frac": f"{tilt:.2f}",
                "detected": "YES" if found else "NO",
                "reproj_err_px": err_str,
            }
        )
    _print_table("Perspective sweep (top-edge tilt fraction of width)", rows)
    return rows


def sweep_noise() -> list[dict]:
    rows = []
    base = _base_scene()
    det = _make_detector()
    for sigma in [0, 5, 10, 20, 40]:
        img = _apply_noise(base, sigma)
        found = _detect(img, det)
        rows.append({"noise_sigma": sigma, "detected": "YES" if found else "NO"})
    _print_table("Noise sweep (additive Gaussian sigma, 0-255 range)", rows)
    return rows


def sweep_scale() -> list[dict]:
    rows = []
    canvas_px = _CANVAS_SIZE
    for marker_px in [200, 120, 80, 50, 30]:
        img = _base_scene(marker_px=marker_px, canvas_px=canvas_px)
        found = _detect(img)
        rows.append({"marker_px": marker_px, "detected": "YES" if found else "NO"})
    _print_table("Scale sweep (marker size in pixels, canvas 640x640)", rows)
    return rows


# ---------------------------------------------------------------------------
# Real-image mode
# ---------------------------------------------------------------------------


def run_real_images(real_dir: str | Path) -> list[dict]:
    from benchmarks._real import iter_real_images  # noqa: PLC0415

    rows = []
    total = 0
    found_count = 0
    try:
        for img, img_dict, expected_ids in iter_real_images(real_dir):
            total += 1
            detector = nf.ArucoDetector(img_dict, max_attempts=_ATTEMPTS)
            result = detector.detect(img)
            detected_ids = list(result.ids)
            correct = all(eid in detected_ids for eid in expected_ids)
            found_count += int(correct)
            rows.append(
                {
                    "expected": str(expected_ids),
                    "detected": str(detected_ids),
                    "correct": "YES" if correct else "NO",
                }
            )
    except FileNotFoundError as exc:
        print(f"\n[real-images] {exc}")
        return []

    _print_table(
        f"Real images — {found_count}/{total} correct ({100 * found_count / max(total, 1):.0f}%)",
        rows,
    )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure nanofractal detection robustness under synthetic degradations."
    )
    parser.add_argument(
        "--real-dir",
        metavar="DIR",
        default=None,
        help="Run real-image evaluation from DIR (needs DIR/manifest.json).",
    )
    parser.add_argument("--json", metavar="PATH", help="Write JSON results to PATH")
    args = parser.parse_args()

    print(f"\nnf version : {nf.__version__}")
    print(f"cv2 available: {_HAS_CV2}" + (f"  (v{cv2.__version__})" if _HAS_CV2 else ""))
    print(f"dict: {_DICT.name}  marker_id: {_MARKER_ID}  max_attempts: {_ATTEMPTS}")

    results: dict[str, Any] = {}

    results["blur"] = sweep_blur()
    results["rotation"] = sweep_rotation()
    results["perspective"] = sweep_perspective()
    results["noise"] = sweep_noise()
    results["scale"] = sweep_scale()

    if args.real_dir:
        results["real_images"] = run_real_images(args.real_dir)

    print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"JSON results written to {args.json}")


if __name__ == "__main__":
    main()
