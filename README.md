# nanofractal

[![codecov](https://codecov.io/gh/Onikore/nanofractal/branch/main/graph/badge.svg)](https://codecov.io/gh/Onikore/nanofractal)

Fast **fractal fiducial-marker** detection for Python. Fractal markers are
nested square markers: they stay detectable under heavy occlusion and over a
wide range of distances, and every visible level contributes corner
correspondences for accurate pose.

`nanofractal` wraps an optimised C++ pipeline with [nanobind](https://github.com/wjakob/nanobind):
**zero-copy** NumPy ↔ `cv::Mat`, the **GIL is released** during detection,
**multi-threaded** single-frame detection, and a **parallel batch** API.

Since 0.6 the library is fractal-only (the ArUco detector was removed).

**📖 [User guide](https://github.com/Onikore/nanofractal/tree/main/guide)** —
markers and printing, detection, pose, video, **how to use it most
efficiently**, parameter tuning, API reference, troubleshooting. Every example
in it is executed by the test-suite.

---

## Performance

The hot path (adaptive threshold, contour tracing, polygon fitting, decoding)
is a purpose-built reimplementation that returns **bit-identical outer
corners** to the reference detector (checked by the test-suite against
`_reference_detect`), and runs on all cores *within* a single frame.

Median latency per frame, `FRACTAL_5L_6`, synthetic scenes from
`nanofractal.bench`, i5-12400 (6 cores / 12 threads), `NF_MARCH=x86-64-v3`,
**OpenCV 4.10 — the configuration of the published wheels**:

| scene / mode | res | 0.5, 1 core | 0.6, 1 core | 0.5, all cores | 0.6, all cores | gain (all cores) |
|---|---|---:|---:|---:|---:|---:|
| marker in clutter, `detect` | 1080p | 11.3 ms | 3.18 ms | 12.3 ms | **1.14 ms** | ×10.8 |
| marker in clutter, `with_inner_points` | 1080p | 13.4 ms | 4.23 ms | 14.4 ms | **2.29 ms** | ×6.3 |
| no marker (clutter only) | 1080p | 12.7 ms | 2.75 ms | 13.8 ms | **0.71 ms** | ×19 |
| marker in clutter, `detect` | 4K | 52.6 ms | 12.5 ms | 49.5 ms | **4.17 ms** | ×11.9 |
| marker in clutter, `with_inner_points` | 4K | 54.6 ms | 16.3 ms | 55.3 ms | **7.40 ms** | ×7.5 |
| no marker (clutter only) | 4K | 55.3 ms | 12.7 ms | 55.1 ms | **4.40 ms** | ×12.5 |
| `detect_batch`, 48 × 1080p | — | — | — | 382 fps | **~1360 fps** | ×3.6 |

With OpenCV 5.0 (whose own `findContours` is already ~3× faster than 4.10's)
0.6 is 2–2.5× faster than 0.5 on one core and 3–8.5× with all cores, e.g.
1080p `detect` 4.66 → 0.92 ms and 4K 15.9 → 4.14 ms.

Reproduce on your machine:

```bash
python -m nanofractal.bench --resolution 1920x1080 --scene perspective
python benchmarks/fractal_bench.py run --resolutions 1920x1080 3840x2160 --batch
```

Levers:

- **`nf.set_num_threads(n)`** — threads used inside one `detect()` call
  (default: all cores; `1` = sequential, best for many concurrent detectors).
  Next to video capture/encoding, fewer threads are often faster (Raspberry
  Pi 5 with a 60 fps camera and preview: 2 threads 57.5 frames/s, 4 threads
  32.5) — start with half the cores and measure the whole pipeline.
- **`params.detection_scale`** — runs the threshold/contour stage on a
  downscaled frame (corners are still refined at full resolution); `0.5` is
  roughly 3–4× faster at 4K.
- **`FractalTracker`** — for video: searches near the previous frame's
  markers (whole frame when they are lost), with identical accuracy; ~2×
  faster at 4K and 1.4–1.8× at 1080p on a moving-marker sequence.
- **`detect_batch`** — parallelises across frames for offline throughput.

---

## Installation

```bash
pip install nanofractal
```

Wheels are available for Linux x86_64/aarch64, macOS arm64/x86_64 (macOS 11+)
and Windows AMD64, CPython 3.9–3.14 (x86_64 wheels require AVX2, i.e. Haswell
2013 or newer). They bundle a minimal OpenCV, so no
system OpenCV is required at runtime.

### Build from source

You need a C++17 compiler, CMake ≥ 3.18 and a development OpenCV — 4.x
(`core`, `imgproc`, `calib3d`, `features2d`) or 5.x (`core`, `imgproc`,
`geometry`, `features`):

```bash
sudo apt-get install -y build-essential cmake libopencv-dev   # Debian/Ubuntu
pip install .
NF_MARCH=x86-64-v3 pip install .     # AVX2 build (what the x86_64 wheels use)
```

---

## Quick start

Inputs are plain NumPy `uint8` arrays — `(H, W)` / `(H, W, 1)` grayscale or
`(H, W, 3)` BGR, **C-contiguous**.

### Generate a marker

```python
import nanofractal as nf

marker = nf.generate_fractal("FRACTAL_5L_6", size_px=1200)   # every nested level
outer = nf.generate_fractal("FRACTAL_5L_6", size_px=1200, nested=False)  # external level only
```

The image spans the marker's black border exactly; leave a white quiet zone
around it when printing. Configs: `FRACTAL_2L_6`, `FRACTAL_3L_6`,
`FRACTAL_4L_6`, `FRACTAL_5L_6` (number of nested levels).

### Detect

```python
import cv2
import nanofractal as nf

image = cv2.imread("scene.png")                  # (H, W, 3) uint8 BGR
det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.2)  # side in metres

res = det.detect(image)
print(res.ids)       # int32 (N,)      — one entry per visible marker level
print(res.corners)   # float32 (N,4,2) — outer corners, sub-pixel
for level_id, corners in res:
    ...
```

### Pose (occlusion-robust)

`estimate_pose` returns `(rvec, tvec, reproj_err)` or `None`. It uses every
visible inner **and** outer corner correspondence when available and falls back
to the outer corners of every detected level otherwise (each at its true size,
so the scale is right even when only an inner level is visible); `reproj_err` (RMS px) lets you gate noisy
poses.

```python
res = det.detect(image, with_inner_points=True)
pose = det.estimate_pose(res, camera_matrix, dist_coeffs)       # fisheye=True for the fisheye model
if pose is not None:
    rvec, tvec, reproj_err = pose
    det.draw(image, res, camera_matrix, dist_coeffs, rvec, tvec)   # outlines + axes
```

The raw correspondences are exposed too: `res.points_2d` float32 `(M, 2)` and
`res.points_3d` float32 `(M, 3)` (planar, z = 0).

### Tune detection

```python
p = nf.DetectorParams()
p.detection_scale = 0.5      # downscaled threshold/contour stage
p.min_contour_size = 60      # detect smaller markers (default 120 px perimeter)
p.adaptive_block_size = 21   # threshold window (odd, >= 3; -1 = auto)
p.adaptive_c = 7.0           # threshold constant
p.approx_poly_rate = 0.05    # polygon fit epsilon = perimeter * rate
p.subpix_win_size = 4        # corner refinement half-window (0 = off)
p.kfilter_min_dist = 10.0    # min distance between inner-corner keypoints
det = nf.FractalDetector("FRACTAL_5L_6", params=p)
det.params.adaptive_c = 5.0  # changes apply to the next detect()
```

### Video

```python
tracker = nf.FractalTracker(det)          # margin=0.25, pad=16
for frame in frames:
    res = tracker.detect(frame)           # same results as det.detect(frame)
```

### More

- `detect(image, roi=(x, y, w, h))` — detect in a zero-copy sub-rectangle;
  corners come back in full-image coordinates.
- `detect_batch(images, num_threads=0, roi=None)` — parallel over frames.
- `refine_corners(image, corners, win_size=5)` — standalone `cornerSubPix`.
- `PoseSmoother(process_noise, measurement_noise, mode="kalman"|"ema")` —
  temporal pose smoothing.
- `draw(..., inplace=False)` — draw on a copy.

Errors: wrong dtype / non-contiguous input → `TypeError`; unsupported shape,
empty frame, invalid config or parameters, invalid intrinsics → `ValueError`.

---

## How it is fast

- **Fused adaptive threshold.** Separable box sums (horizontal first, uint16,
  ring buffer + running vertical window), reproducing OpenCV's fixed-point /
  float rounding of the mean exactly, written straight into a zero-bordered
  buffer together with 8-neighbour occupancy masks. Row stripes run in parallel.
- **Contour tracing.** Suzuki–Abe border following identical to
  `cv::findContours(RETR_LIST, CHAIN_APPROX_NONE)` (sequential), with SIMD
  start detection, one mask load per step and no allocation for contours below
  the minimum size. Isolated speckle pixels (most contours in noisy frames) are
  skipped without tracing. Row bands are traced in parallel: each band keeps
  its own pixel labels and a contour is kept by the band holding its start,
  the ownership rule also used by OpenCV's data-race-free TRUCO tracer; the
  test-suite checks the result against `findContours` on millions of
  contours, including bands one row high. It is 2–3× faster than OpenCV 5.0's
  own parallel `findContours` at every thread count (1080p, 12 threads:
  0.31 vs 0.72 ms).
- **Polygon fit + decoding** in parallel per contour, with exact replicas of
  OpenCV 4's and OpenCV 5's `approxPolyDP` (they differ) and no per-frame heap
  traffic.
- **Inner points.** FAST runs only around the marker; keypoint filtering and
  matching use grids instead of O(n²) scans.
- **Tracking** (`FractalTracker`): region-of-interest search around the
  previous detection, after Romero-Ramirez et al., *Speeded up detection of
  squared fiducial markers* (2018). Their other lever — shrinking the image to
  the previous marker size — was tested and rejected: it raised the corner
  error from 0.39 to 1.3 px.
- **Zero-copy input, GIL released**, a detector pool for `detect_batch`.

The one intentional behaviour difference from the reference detector: with
`with_inner_points=True`, the FAST response threshold is computed over the
marker's neighbourhood instead of the whole frame. Outer corners are unchanged.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use this in research, please cite the original work:

- F. J. Romero-Ramirez, R. Muñoz-Salinas, R. Medina-Carnicer,
  *"Fractal Markers: A New Approach for Long-Range Marker Pose Estimation Under
  Occlusion"*, IEEE Access, 7, 2019.
- F. J. Romero-Ramirez, R. Muñoz-Salinas, R. Medina-Carnicer,
  *"Speeded up detection of squared fiducial markers"*, Image and Vision
  Computing, 76, 2018.

## License

Apache-2.0. The vendored fractal detector is © its authors and used under its
terms; see `third_party/` and `PATCHES.md`.
