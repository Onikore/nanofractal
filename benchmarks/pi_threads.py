"""benchmarks/pi_threads.py — how detect() scales with threads, per frame size.

On a Raspberry Pi 5 (0.7.0) four threads were no faster than one from
1280x720 up, while 640x480 scaled ×2.5. Two explanations predict different
curves:

* cache: the frame and its threshold buffers (about 4 bytes per pixel)
  outgrow the 2 MB shared L3 near 800x600 — scaling then breaks at that size
  for every thread count, and the spread (median / min) stays small;
* a busy core (camera, desktop, another process): only the run that uses
  every core suffers, at every size, with a large median / min spread.

The script prints, for each size and thread count, the median and min time
of ``detect`` and the speedup over one thread, and the load average before
and after. Run it with nothing else running, then again with the camera
pipeline running.

    python benchmarks/pi_threads.py --json pi_threads.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import nanofractal as nf
from nanofractal.bench import generate_scene

SIZES = ("640x480", "800x600", "960x720", "1024x768", "1280x720", "1456x1088", "1920x1080")


def _time(det, frame, min_time):
    det.detect(frame)
    ts = []
    t_end = time.perf_counter() + min_time
    while time.perf_counter() < t_end or len(ts) < 20:
        t0 = time.perf_counter()
        det.detect(frame)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.min(ts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", nargs="+", default=list(SIZES))
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--scenes", nargs="+", default=["clutter", "perspective"])
    ap.add_argument("--min-time", type=float, default=2.0, help="seconds per cell")
    ap.add_argument("--json")
    args = ap.parse_args()

    load0 = os.getloadavg() if hasattr(os, "getloadavg") else None
    det = nf.FractalDetector("FRACTAL_5L_6")
    rows = []
    for scene in args.scenes:
        for size in args.sizes:
            w, h = (int(v) for v in size.split("x"))
            frame = generate_scene(w, h, scene, "FRACTAL_5L_6")
            base = None
            line = f"{scene:12s} {size:>10s}"
            for t in args.threads:
                nf.set_num_threads(t)
                med, mn = _time(det, frame, args.min_time)
                base = base or med
                rows.append(
                    {"scene": scene, "size": size, "threads": t, "median_ms": med, "min_ms": mn}
                )
                line += f" | {t}t {med:6.2f} ms x{base / med:4.2f} spread {med / mn:4.2f}"
            print(line, flush=True)
    load1 = os.getloadavg() if hasattr(os, "getloadavg") else None
    print(f"load average before {load0}, after {load1}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"version": nf.__version__, "load": [load0, load1], "rows": rows}, f, indent=1
            )


if __name__ == "__main__":
    main()
