"""benchmarks/frames_report.py — one CSV row per real frame (RPI_TEST_PLAN).

Runs ``detect``, ``detect(all_instances=True)`` and ``diagnose`` on every
image of a folder and writes what was found and, when nothing or not every
level was found, why: the closest rejected quad (wrong border cells, wrong
bits, cell contrast). With a calibration (``--calib calib.npz`` holding
``K`` and ``dist``; ``--fisheye`` for the fisheye model) it also writes the
pose, its standard deviations from ``Pose.cov`` and the error of the other
planar solution. Every row also has the reading quality of the largest level
found (``cell_px``, ``contrast``, ``bit_errors``). ``--prior`` treats the
folder as a video (files in name order) and passes the last non-empty result
as ``prior``, like ``FractalTracker(use_prior=True)``; ``--capture N`` runs a
``FractalTracker(capture_frames=N)`` instead (picks up the marker before it is
read; implies ``--prior``).

    python benchmarks/frames_report.py frames/ --config FRACTAL_3L_6 \\
        --size 0.18 --calib calib.npz --csv report.csv

The summary printed at the end gives the share of frames with any level and
with the outer level (id 0), and the counts of dropped (``inconsistent_ids``)
levels and of frames with several copies of the marker.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import nanofractal as nf

_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def _near_miss(diag):
    """The rejected quad closest to a valid code: (border, bits, contrast, side px)."""
    best = None
    for q in diag.quads:
        # flat blobs (contrast < 20 grey levels) are not near misses
        if q.reason not in ("border", "bits") or q.contrast < 20:
            continue
        key = (q.border_errors, q.bit_errors, -q.contrast)
        if best is None or key < best[0]:
            side = float(np.linalg.norm(q.corners - np.roll(q.corners, 1, 0), axis=1).mean())
            best = (key, q, side)
    if best is None:
        return "", "", "", ""
    _, q, side = best
    return q.border_errors, q.bit_errors, round(q.contrast, 1), round(side, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--config", required=True)
    ap.add_argument("--size", type=float, default=-1.0, help="marker side in metres")
    ap.add_argument("--calib", help="npz with K and dist (adds pose columns)")
    ap.add_argument("--fisheye", action="store_true")
    ap.add_argument("--inner", action="store_true", help="pose with inner points")
    ap.add_argument(
        "--prior", action="store_true", help="frames are a video: read with the last result"
    )
    ap.add_argument(
        "--capture", type=int, default=0, help="video: FractalTracker(capture_frames=N)"
    )
    ap.add_argument("--csv", default="report.csv")
    args = ap.parse_args()

    det = nf.FractalDetector(args.config, marker_size=args.size)
    K = dist = None
    if args.calib:
        c = np.load(args.calib)
        K, dist = c["K"], c["dist"]

    files = sorted(p for p in Path(args.folder).rglob("*") if p.suffix.lower() in _EXT)
    if not files:
        raise SystemExit(f"no images in {args.folder}")
    cols = [
        "file",
        "ids",
        "inconsistent_ids",
        "multiple_instances",
        "n_instances",
        "detect_ms",
        "contours",
        "not_quad",
        "accepted",
        "duplicate",
        "border",
        "bits",
        "inconsistent",
        "miss_border",
        "miss_bits",
        "miss_contrast",
        "miss_side_px",
        "method",
        "n_points",
        "reproj_err",
        "reproj_err_alt",
        "tx",
        "ty",
        "tz",
        "sd_tx",
        "sd_ty",
        "sd_tz",
        "sd_rot_deg",
        "cell_px",
        "contrast",
        "bit_errors",
    ]
    det.detect(np.zeros((64, 64), np.uint8))  # warm-up: first call allocates
    tracker = nf.FractalTracker(det, capture_frames=args.capture) if args.capture else None
    n_any = n_outer = n_drop = n_multi = n_prior = 0
    last = None
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"skip (unreadable): {p}")
                continue
            t0 = time.perf_counter()
            if tracker is not None:
                res = tracker.detect(img, with_inner_points=args.inner)
            else:
                res = det.detect(img, with_inner_points=args.inner, prior=last)
            if args.prior and len(res):
                last = res
            ms = (time.perf_counter() - t0) * 1e3
            allr = det.detect(img, all_instances=True)
            diag = det.diagnose(img)
            counts = {r: 0 for r in ("accepted", "duplicate", "border", "bits", "inconsistent")}
            for q in diag.quads:
                counts[q.reason] += 1
            row = [
                p.name,
                " ".join(map(str, res.ids.tolist())),
                " ".join(map(str, res.inconsistent_ids.tolist())),
                int(res.multiple_instances),
                allr.n_instances,
                round(ms, 2),
                diag.n_contours,
                diag.n_not_quad,
                *counts.values(),
                *_near_miss(diag),
            ]
            pose_cols = [""] * 11
            if K is not None and len(res):
                pose = det.estimate_pose(res, K, dist, fisheye=args.fisheye, full=True)
                if pose is not None:
                    sd = [""] * 4
                    if pose.cov is not None:
                        d = np.sqrt(np.diag(pose.cov))
                        rot = float(np.degrees(np.linalg.norm(d[:3])))
                        sd = [*np.round(d[3:], 5), round(rot, 3)]
                    alt = "" if pose.reproj_err_alt is None else round(pose.reproj_err_alt, 3)
                    pose_cols = [
                        pose.method,
                        pose.n_points,
                        round(pose.reproj_err, 3),
                        alt,
                        *np.round(pose.tvec, 4),
                        *sd,
                    ]
            qual = [""] * 3
            if len(res):
                k = int(np.argmax(res.cell_px))
                qual = [round(float(res.cell_px[k]), 2), round(float(res.contrast[k]), 1)]
                qual.append(int(res.bit_errors[k]))
            w.writerow(row + pose_cols + qual)
            n_prior += bool(len(res)) and int(res.bit_errors.max()) > 0
            n_any += len(res) > 0
            n_outer += 0 in res.ids
            n_drop += len(res.inconsistent_ids) > 0
            n_multi += allr.n_instances > 1
    n = len(files)
    print(
        f"{n} frames: any level {n_any / n:.1%}, outer level {n_outer / n:.1%}, "
        f"frames with dropped levels {n_drop}, with several copies {n_multi}, "
        f"read through the prior {n_prior} -> {args.csv}"
    )


if __name__ == "__main__":
    main()
