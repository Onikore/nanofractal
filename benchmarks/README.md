# benchmarks/

Standalone benchmark and quality scripts for `nanofractal`. They do **not**
require pytest and are excluded from the test suite (`testpaths = ["tests"]`).
Scenes come from `nanofractal.bench.generate_scene` (numpy only, seeded), so
results are reproducible across machines and builds.

---

## fractal_bench.py — latency matrix and output equivalence

Times every (scene × resolution × mode) cell and can dump / compare the
detections themselves, which is how performance changes are checked for
identical output.

```
python benchmarks/fractal_bench.py run                       # all scenes, 480p..4K, 3 modes
python benchmarks/fractal_bench.py run --resolutions 1920x1080 3840x2160 --batch
python benchmarks/fractal_bench.py run --threads 1 --json new.json --vs base.json
python benchmarks/fractal_bench.py dump --out ref.npz        # on a reference build
python benchmarks/fractal_bench.py compare ref.npz           # exit 1 on any difference
```

| flag | meaning |
|------|---------|
| `--scenes` | `clean` (frontal marker), `perspective` (marker in clutter), `clutter` (no marker) |
| `--modes` | `detect`, `inner` (`with_inner_points=True`), `scale05` (`detection_scale=0.5`) |
| `--resolutions` | `WxH` list (default 640x480 … 3840x2160) |
| `--threads N` | threads per `detect()` (`nf.set_num_threads`; 0 = all cores) |
| `--batch` | also report `detect_batch` throughput (48 × 1080p, all cores) |
| `--json / --vs` | write results / print speedups against a previous JSON |
| `--atol` | (`compare`) corner tolerance, default 1e-3 px |

Single-frame CLI: `python -m nanofractal.bench --resolution 1920x1080 --scene perspective --mode inner`.

---

## robustness.py — detection quality under synthetic degradations

For each degradation level reports which nested levels of `FRACTAL_5L_6` were
found, how many inner-corner correspondences were matched and the RMS
reprojection error of the pose. Deterministic, so two runs can be diffed
(`--json`).

```
python benchmarks/robustness.py
python benchmarks/robustness.py --json results.json
python benchmarks/robustness.py --real-dir benchmarks/real_images
```

| sweep | values |
|-------|--------|
| **blur** | Gaussian σ ∈ {0, 1, 2, 3, 5} px |
| **noise** | additive Gaussian σ ∈ {0, 5, 10, 20, 40} |
| **perspective** | marker tilted about the x axis ∈ {0°, 15°, 30°, 45°, 60°} (physically valid views) |
| **rotation** | in-plane ∈ {0°, 5°, 15°, 30°, 45°} |
| **scale** | marker side ∈ {1100, 780, 400, 200, 120} px on a 1280×1280 canvas |

---

## reading.py — far markers, shadows, marker-like negatives

The bit reader under its hardest synthetic cases (`FRACTAL_4L_6`,
1280×720 frames on clutter, blur 0.6, noise 3, random pose): share of frames
where the external level is found at 2.2–4.4 px per cell, under a hard
shadow edge (×0.6, ×0.45, ×0.3) and with 10–40 % of the marker covered by a
disc (also: any level found), plus how many of 300 frames full of
marker-like objects (random codes with a black border, markers of other
configurations) produce any detection — that must stay 0.

```
python benchmarks/reading.py
python benchmarks/reading.py --json results.json
```

---

## Real-image slot (`benchmarks/real_images/`)

Drop your own fractal-marker captures into `benchmarks/real_images/` with a
`manifest.json` (copy `manifest.example.json`) and run
`python benchmarks/robustness.py --real-dir benchmarks/real_images`.
The real-image loader needs `opencv-python` (`pip install -e ".[bench]"`).
No real images are shipped with the library.
