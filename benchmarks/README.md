# benchmarks/

Standalone benchmark scripts for `nanofractal`.  They do **not** require pytest
and are excluded from the test suite (`testpaths = ["tests"]` in `pyproject.toml`).

---

## compare_opencv.py — speed comparison vs cv2.aruco

Measures per-frame detection latency (mean / median / p95) and FPS for both
`nanofractal` and `cv2.aruco` on synthetic scenes at configurable resolutions
and dictionaries, then prints a side-by-side table.

```
python benchmarks/compare_opencv.py
python benchmarks/compare_opencv.py \
    --resolutions 640x480,1280x720,1920x1080 \
    --dicts DICT_4X4_50,DICT_5X5_100 \
    --frames 200 \
    --json results.json
```

| flag | default | meaning |
|------|---------|---------|
| `--resolutions` | `640x480,1280x720,1920x1080` | comma-separated `WxH` list |
| `--dicts` | `DICT_4X4_50,DICT_5X5_100` | `nf.Dict` names |
| `--frames` | `100` | timing iterations per cell |
| `--json PATH` | — | write JSON output to PATH |

If `cv2` is not installed the script exits cleanly with instructions.

---

## robustness.py — detection rate under synthetic degradations

Sweeps five degradation axes using a synthetic ArUco scene and reports detection
rate.  Where a known geometric transform is applied (rotation / perspective),
reprojection error is also reported.

```
python benchmarks/robustness.py
python benchmarks/robustness.py --json results.json
python benchmarks/robustness.py --real-dir benchmarks/real_images
```

Sweeps:

| sweep | values |
|-------|--------|
| **blur** | GaussianBlur σ ∈ {0, 1, 2, 3, 5} |
| **rotation** | in-plane ∈ {0°, 5°, 15°, 30°, 45°} |
| **perspective** | top-edge tilt ∈ {0.00, 0.05, 0.10, 0.20, 0.30} of width |
| **noise** | additive Gaussian σ ∈ {0, 5, 10, 20, 40} |
| **scale** | marker size ∈ {200, 120, 80, 50, 30} px on a 640×640 canvas |

The `--real-dir` flag enables real-image evaluation (see below).

---

## Real-image slot (`benchmarks/real_images/`)

Drop your own ArUco captures into `benchmarks/real_images/` together with a
`manifest.json` (copy `manifest.example.json` as a starting point) and run:

```
python benchmarks/robustness.py --real-dir benchmarks/real_images
```

No real images are shipped with the library.

---

## [bench] extra — cv2 dependency

`compare_opencv.py` and the image-ops parts of `robustness.py` require
`opencv-python`.  Install it via the optional extra:

```
pip install -e ".[bench]"
```

or directly:

```
pip install opencv-python
```

If `cv2` is absent, all cv2-dependent sweeps are skipped gracefully and
`compare_opencv.py` exits with an informative message.
