"""benchmarks/fractal_bench.py — latency matrix + output-equivalence checks.

Subcommands:
    run      time every (scene, resolution, mode) cell; print a table, optional --json
    dump     save the detections of every cell to an .npz (reference outputs)
    compare  re-run every cell and diff against a dumped .npz (ids exact,
             corners / inner points within --atol)

Frames come from nanofractal.bench.generate_scene (seeded, numpy only), so the
same cells are reproduced exactly across builds and machines. Typical loop:

    python benchmarks/fractal_bench.py dump --out ref.npz      # on the baseline build
    python benchmarks/fractal_bench.py run --json base.json
    ... optimise, rebuild ...
    python benchmarks/fractal_bench.py compare ref.npz
    python benchmarks/fractal_bench.py run --json new.json --vs base.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np

import nanofractal as nf
from nanofractal.bench import MODES, SCENES, generate_scene, make_detector, run_once

RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
CONFIG = "FRACTAL_5L_6"


def _cells(scenes, resolutions, modes):
    for scene in scenes:
        for w, h in resolutions:
            for mode in modes:
                yield scene, w, h, mode


def _key(scene, w, h, mode):
    return f"{scene}_{w}x{h}_{mode}"


_FRAMES: dict = {}


def _frame(scene, w, h):
    k = (scene, w, h)
    if k not in _FRAMES:
        _FRAMES[k] = generate_scene(w, h, scene, CONFIG, seed=0)
    return _FRAMES[k]


def time_cell(det, frame, mode, min_time=1.0, min_iters=15, max_iters=2000):
    for _ in range(3):
        run_once(det, frame, mode)
    lat = []
    t_start = time.perf_counter()
    while len(lat) < max_iters:
        t0 = time.perf_counter()
        run_once(det, frame, mode)
        lat.append((time.perf_counter() - t0) * 1e3)
        if len(lat) >= min_iters and time.perf_counter() - t_start > min_time:
            break
    return {"median_ms": statistics.median(lat), "min_ms": min(lat), "iters": len(lat)}


def cmd_run(args):
    if hasattr(nf, "set_num_threads"):  # (absent in old builds benchmarked for comparison)
        nf.set_num_threads(args.threads)
    rows = []
    prev = {}
    if args.vs:
        with open(args.vs) as fh:
            prev = {r["key"]: r for r in json.load(fh)["rows"]}
    print(
        f"nanofractal {nf.__version__}  OpenCV {nf._opencv_version()}  {platform.machine()}"
        f"  threads/detect={nf.get_num_threads() if hasattr(nf, 'get_num_threads') else '?'}"
    )
    hdr = f"{'cell':34s} {'median ms':>10s} {'min ms':>9s} {'iters':>6s}  found"
    if prev:
        hdr += "      vs base"
    print(hdr)
    for scene, w, h, mode in _cells(args.scenes, args.resolutions, args.modes):
        det = make_detector(mode, CONFIG)
        frame = _frame(scene, w, h)
        r = time_cell(det, frame, mode, args.min_time)
        r["key"] = _key(scene, w, h, mode)
        r["found"] = sorted(run_once(det, frame, mode).ids.tolist())
        rows.append(r)
        line = (
            f"{r['key']:34s} {r['median_ms']:10.3f} {r['min_ms']:9.3f} {r['iters']:6d}  "
            f"{r['found']}"
        )
        if r["key"] in prev:
            line += f"  x{prev[r['key']]['median_ms'] / r['median_ms']:.2f}"
        print(line, flush=True)

    if args.batch:
        imgs = [generate_scene(1920, 1080, "perspective", CONFIG, seed=i) for i in range(48)]
        det = make_detector("detect", CONFIG)
        det.detect_batch(imgs, num_threads=0)
        best = min(
            (lambda t0: (det.detect_batch(imgs, num_threads=0), time.perf_counter() - t0)[1])(
                time.perf_counter()
            )
            for _ in range(5)
        )
        fps = len(imgs) / best
        print(f"batch 48x1080p perspective, all cores: {fps:.1f} frames/s")
        rows.append({"key": "batch_1080p_fps", "fps": fps, "median_ms": 1e3 / fps})

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"opencv": nf._opencv_version(), "rows": rows}, fh, indent=1)


def _detections(scene, w, h, mode):
    det = make_detector(mode, CONFIG)
    res = run_once(det, _frame(scene, w, h), mode)
    out = {"ids": res.ids, "corners": res.corners}
    if res.points_2d is not None:
        out["p2d"] = res.points_2d
        out["p3d"] = res.points_3d
    return out


def cmd_dump(args):
    data = {}
    for cell in _cells(args.scenes, args.resolutions, args.modes):
        for name, arr in _detections(*cell).items():
            data[f"{_key(*cell)}/{name}"] = arr
    np.savez_compressed(args.out, **data)
    print(f"wrote {len(data)} arrays to {args.out}")


def cmd_compare(args):
    ref = np.load(args.ref)
    worst = 0.0
    bad = 0
    for cell in _cells(args.scenes, args.resolutions, args.modes):
        key = _key(*cell)
        got = _detections(*cell)
        msgs = []
        if not np.array_equal(got["ids"], ref[f"{key}/ids"]):
            msgs.append(f"ids {ref[f'{key}/ids'].tolist()} -> {got['ids'].tolist()}")
        else:
            d = float(np.abs(got["corners"] - ref[f"{key}/corners"]).max(initial=0))
            worst = max(worst, d)
            if d > args.atol:
                msgs.append(f"corners max diff {d:.4g}")
        if "p2d" in got:
            r2, r3 = ref[f"{key}/p2d"], ref[f"{key}/p3d"]
            if got["p2d"].shape != r2.shape or not np.array_equal(got["p3d"], r3):
                msgs.append(f"inner points {len(r2)} -> {len(got['p2d'])}")
            else:
                d = float(np.abs(got["p2d"] - r2).max(initial=0))
                worst = max(worst, d)
                if d > args.atol:
                    msgs.append(f"p2d max diff {d:.4g}")
        if msgs:
            bad += 1
            print(f"DIFF {key}: {'; '.join(msgs)}")
    print(f"{'OK' if not bad else 'MISMATCH'}: {bad} differing cells, worst coord diff {worst:.3g}")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "dump", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--scenes", nargs="+", default=list(SCENES), choices=SCENES)
        p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
        p.add_argument(
            "--resolutions",
            nargs="+",
            default=list(RESOLUTIONS),
            type=lambda s: tuple(int(v) for v in s.split("x")),
        )
        if name == "run":
            p.add_argument("--json")
            p.add_argument("--vs", help="previous --json to print speedups against")
            p.add_argument("--min-time", type=float, default=1.0)
            p.add_argument("--batch", action="store_true")
            p.add_argument(
                "--threads",
                type=int,
                default=0,
                help="threads per detect() (nf.set_num_threads); 0 = all cores",
            )
        elif name == "dump":
            p.add_argument("--out", required=True)
        else:
            p.add_argument("ref")
            p.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()
    {"run": cmd_run, "dump": cmd_dump, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
