"""benchmarks/compare_opencv.py — nanofractal vs cv2.aruco side-by-side speed comparison.

Usage:
    python benchmarks/compare_opencv.py
    python benchmarks/compare_opencv.py \
        --resolutions 640x480,1280x720 --dicts DICT_4X4_50 --frames 50
    python benchmarks/compare_opencv.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np

import nanofractal as nf

# ---------------------------------------------------------------------------
# cv2 import — soft dependency
# ---------------------------------------------------------------------------
try:
    import cv2  # noqa: F401

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ---------------------------------------------------------------------------
# Dict name → cv2.aruco constant
# ---------------------------------------------------------------------------
_NF_TO_CV2_NAME: dict[str, str] = {
    "DICT_4X4_50": "DICT_4X4_50",
    "DICT_4X4_100": "DICT_4X4_100",
    "DICT_4X4_250": "DICT_4X4_250",
    "DICT_4X4_1000": "DICT_4X4_1000",
    "DICT_5X5_50": "DICT_5X5_50",
    "DICT_5X5_100": "DICT_5X5_100",
    "DICT_5X5_250": "DICT_5X5_250",
    "DICT_5X5_1000": "DICT_5X5_1000",
    "DICT_6X6_50": "DICT_6X6_50",
    "DICT_6X6_100": "DICT_6X6_100",
    "DICT_6X6_250": "DICT_6X6_250",
    "DICT_6X6_1000": "DICT_6X6_1000",
    "DICT_7X7_50": "DICT_7X7_50",
    "DICT_7X7_100": "DICT_7X7_100",
    "DICT_7X7_250": "DICT_7X7_250",
    "DICT_7X7_1000": "DICT_7X7_1000",
}


def _resolve_nf_dict(name: str) -> nf.Dict:
    try:
        return nf.Dict[name]
    except KeyError:
        sys.exit(f"Unknown nanofractal dictionary: {name!r}. Valid names: {list(_NF_TO_CV2_NAME)}")


def _resolve_cv2_dict(name: str) -> Any:
    """Return the cv2.aruco dict constant for *name*, or None if unavailable."""
    if not _HAS_CV2:
        return None
    cv2_name = _NF_TO_CV2_NAME.get(name)
    if cv2_name is None:
        return None
    const = getattr(cv2.aruco, cv2_name, None)
    if const is None:
        return None
    return const


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

_MARKER_ID = 0


def _build_scene(width: int, height: int, nf_dict: nf.Dict) -> np.ndarray:
    """White canvas with a single marker pasted with a quiet-zone margin."""
    marker_size = max(30, min(height, width) // 4)
    # clamp so the marker fits with at least 1-pixel margin on each side
    marker_size = min(marker_size, width - 2, height - 2)

    marker = nf.generate_aruco(_MARKER_ID, size_px=marker_size, dictionary=nf_dict, border_bits=1)

    canvas = np.full((height, width), 255, dtype=np.uint8)
    x0 = (width - marker_size) // 2
    y0 = (height - marker_size) // 2
    canvas[y0 : y0 + marker_size, x0 : x0 + marker_size] = marker
    return canvas


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_nf(frame: np.ndarray, nf_dict: nf.Dict, n: int) -> tuple[np.ndarray, bool]:
    """Time nanofractal detection for *n* frames; return (latencies_ms, detected)."""
    detector = nf.ArucoDetector(nf_dict, max_attempts=1)
    latencies = np.empty(n)
    detected = False
    for i in range(n):
        t0 = time.perf_counter()
        result = detector.detect(frame)
        latencies[i] = (time.perf_counter() - t0) * 1e3
        if _MARKER_ID in result.ids:
            detected = True
    return latencies, detected


def _time_cv2(frame: np.ndarray, cv2_dict_const: Any, n: int) -> tuple[np.ndarray, bool]:
    """Time cv2.aruco detection for *n* frames; return (latencies_ms, detected)."""
    import cv2  # already imported at module level check

    dictionary = cv2.aruco.getPredefinedDictionary(cv2_dict_const)
    params = cv2.aruco.DetectorParameters()

    # cv2.aruco.ArucoDetector is available in OpenCV 4.7+
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)

        def _detect(img: np.ndarray) -> bool:
            corners, ids, _ = detector.detectMarkers(img)
            return ids is not None and _MARKER_ID in ids.flatten()

    else:
        # Older API fallback
        def _detect(img: np.ndarray) -> bool:  # type: ignore[misc]
            _, ids, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=params)
            return ids is not None and _MARKER_ID in ids.flatten()

    latencies = np.empty(n)
    detected = False
    for i in range(n):
        t0 = time.perf_counter()
        found = _detect(frame)
        latencies[i] = (time.perf_counter() - t0) * 1e3
        if found:
            detected = True
    return latencies, detected


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _stats(lat: np.ndarray) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(lat)),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "fps": float(1000.0 / np.mean(lat)),
    }


def _print_row(label: str, stats: dict[str, float], detected: bool) -> None:
    print(
        f"  {label:<14} "
        f"mean={stats['mean_ms']:6.2f}ms  "
        f"median={stats['median_ms']:6.2f}ms  "
        f"p95={stats['p95_ms']:6.2f}ms  "
        f"FPS={stats['fps']:7.1f}  "
        f"detected={'YES' if detected else 'NO ':3}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare nanofractal vs cv2.aruco detection speed."
    )
    parser.add_argument(
        "--resolutions",
        default="640x480,1280x720,1920x1080",
        help="Comma-separated WxH list (default: %(default)s)",
    )
    parser.add_argument(
        "--dicts",
        default="DICT_4X4_50,DICT_5X5_100",
        help="Comma-separated dict names (default: %(default)s)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Frames per benchmark run (default: %(default)s)",
    )
    parser.add_argument("--json", metavar="PATH", help="Write JSON results to PATH")
    args = parser.parse_args()

    if not _HAS_CV2:
        print(
            "opencv-python not installed; cv2.aruco comparison unavailable.\n"
            "Install with:  pip install nanofractal[bench]\n"
            "or:            pip install opencv-python"
        )
        sys.exit(0)

    resolutions = []
    for r in args.resolutions.split(","):
        r = r.strip()
        try:
            w, h = (int(v) for v in r.split("x"))
        except ValueError:
            sys.exit(f"Bad resolution {r!r} — use WxH format, e.g. 640x480")
        resolutions.append((w, h))

    dict_names = [d.strip() for d in args.dicts.split(",")]

    print(f"\nnf version : {nf.__version__}")
    print(f"cv2 version: {cv2.__version__}")
    print(f"frames     : {args.frames}\n")
    print("=" * 78)

    all_results: list[dict] = []

    for dict_name in dict_names:
        nf_dict = _resolve_nf_dict(dict_name)
        cv2_const = _resolve_cv2_dict(dict_name)

        print(f"\nDict: {dict_name}")
        print("-" * 78)

        for w, h in resolutions:
            print(f"  Resolution {w}x{h}:")
            frame = _build_scene(w, h, nf_dict)

            # warm-up (2 calls each)
            _time_nf(frame, nf_dict, 2)
            if cv2_const is not None:
                _time_cv2(frame, cv2_const, 2)

            nf_lat, nf_det = _time_nf(frame, nf_dict, args.frames)
            nf_stats = _stats(nf_lat)
            _print_row("nanofractal", nf_stats, nf_det)

            if cv2_const is not None:
                cv2_lat, cv2_det = _time_cv2(frame, cv2_const, args.frames)
                cv2_stats = _stats(cv2_lat)
                _print_row("cv2.aruco", cv2_stats, cv2_det)

                ratio = (
                    cv2_stats["mean_ms"] / nf_stats["mean_ms"]
                    if nf_stats["mean_ms"] > 0
                    else float("inf")
                )
                print(f"  {'speedup':<14} cv2/nf = {ratio:.2f}x")
            else:
                cv2_stats = {}
                cv2_det = None
                print(f"  {'cv2.aruco':<14} (dict not available in this cv2 build)")

            all_results.append(
                {
                    "dict": dict_name,
                    "resolution": f"{w}x{h}",
                    "nanofractal": {**nf_stats, "detected": nf_det},
                    "cv2_aruco": {**cv2_stats, "detected": cv2_det} if cv2_stats else None,
                }
            )

    print("\n" + "=" * 78)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"JSON results written to {args.json}")


if __name__ == "__main__":
    main()
