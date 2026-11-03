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
0.7 reads far markers the reference detector misses (2.6–3 px per cell),
explains rejected candidates (`diagnose`), drops levels that do not fit the
rest, finds every copy of a marker, reports pose uncertainty, scales the
tracker to the marker size and adds NEON code for aarch64. 0.7.1 finds
markers seen from very close on dark frames (a second threshold pass);
0.7.2 reports the reading quality of every level and keeps a marker that
flickers at the edge of range (`prior` / `FractalTracker(use_prior=True)`).
0.7.3 picks up a marker that is in view but never read yet, once it holds
its place for a few frames (`FractalTracker(capture_frames=3)`).

**📖 [User guide](https://github.com/Onikore/nanofractal/tree/main/guide)** —
markers and printing, detection, pose, video, **how to use it most
efficiently**, parameter tuning, API reference, troubleshooting. Every Python
example in it, and in this README, is executed by the test-suite.

---

## Performance

The hot path (adaptive threshold, contour tracing, polygon fitting, decoding)
is a purpose-built reimplementation: every level the reference detector
finds comes back with **bit-identical corners** (checked by the test-suite
against `_reference_detect`), and it runs on all cores *within* a single
frame.

Median latency per frame, `FRACTAL_5L_6`, synthetic scenes from
`nanofractal.bench`, i5-12400 (6 cores / 12 threads), `NF_MARCH=x86-64-v3`,
**OpenCV 4.10 — the configuration of the published wheels**:

| scene / mode | res | 0.5, 1 core | 0.7.3, 1 core | 0.5, all cores | 0.7.3, all cores | gain (all cores) |
|---|---|---:|---:|---:|---:|---:|
| marker in clutter, `detect` | 1080p | 11.3 ms | 3.21 ms | 12.3 ms | **0.99 ms** | ×12.4 |
| marker in clutter, `with_inner_points` | 1080p | 13.4 ms | 4.50 ms | 14.4 ms | **2.18 ms** | ×6.6 |
| no marker (clutter only) | 1080p | 12.7 ms | 3.42 ms | 13.8 ms | **0.85 ms** | ×16 |
| marker in clutter, `detect` | 4K | 52.6 ms | 12.1 ms | 49.5 ms | **4.69 ms** | ×10.6 |
| marker in clutter, `with_inner_points` | 4K | 54.6 ms | 15.6 ms | 55.3 ms | **7.20 ms** | ×7.7 |
| no marker (clutter only) | 4K | 55.3 ms | 13.3 ms | 55.1 ms | **4.95 ms** | ×11 |
| `detect_batch`, 48 × 1080p | — | — | — | 382 fps | **~1500 fps** | ×3.9 |

0.7.3 measured 2026-11-03 (one core pinned with `taskset`); 0.5 on the same
machine earlier. A frame with no marker costs 0.1–0.7 ms more than in 0.6:
it is thresholded a second time for markers seen from very close
(`params.close_pass = False` skips that).

With OpenCV 5.0 (whose own `findContours` is already ~3× faster than 4.10's)
0.6 is 2–2.5× faster than 0.5 on one core and 3–8.5× with all cores, e.g.
1080p `detect` 4.66 → 0.92 ms and 4K 15.9 → 4.14 ms.

0.7 adds a SIMD prefix-scan threshold (`detect` on one core ×1.17–1.34 at
4K, ×1.05–1.12 at 1080p) and NEON code for aarch64 (Raspberry Pi 5, one
core: 1.5–2× faster than the scalar build). The second bit reading costs
nothing measurable on scenes the reference already reads. Every push is
timed against the previous commit in CI and fails on a slowdown.

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
  faster at 4K and 1.4–1.8× at 1080p on a moving-marker sequence. With
  `auto_scale=True` it also shrinks the searched region to the marker size
  (corners still refined on the full frame): 4K, one core, ×5.7 vs `detect`.
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
import numpy as np
import nanofractal as nf

# a test frame: the marker on a white page (or image = cv2.imread("scene.png"))
image = np.full((900, 1200), 255, np.uint8)
image[150:750, 300:900] = nf.generate_fractal("FRACTAL_5L_6", 600)
det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.2)  # side in metres

res = det.detect(image)
print(res.ids)       # int32 (N,)      — one entry per visible marker level
print(res.corners)   # float32 (N,4,2) — outer corners, sub-pixel
for level_id, corners in res:
    print(level_id, corners[0])
assert 0 in res.ids
```

### Pose (occlusion-robust)

`estimate_pose` returns `(rvec, tvec, reproj_err)` or `None`. It uses every
visible inner **and** outer corner correspondence when available and falls back
to the outer corners of every detected level otherwise (each at its true size,
so the scale is right even when only an inner level is visible); `reproj_err` (RMS px) lets you gate noisy
poses.

```python
camera_matrix = np.array([[1000.0, 0, 600], [0, 1000.0, 450], [0, 0, 1]])
dist_coeffs = np.zeros(5)

res = det.detect(image, with_inner_points=True)
pose = det.estimate_pose(res, camera_matrix, dist_coeffs)       # fisheye=True for the fisheye model
if pose is not None:
    rvec, tvec, reproj_err = pose
    print(tvec.ravel(), reproj_err)    # marker 0.2 m wide, 600 px: about 0.33 m away
    vis = det.draw(image, res, camera_matrix, dist_coeffs, rvec, tvec)   # outlines + axes
```

The raw correspondences are exposed too: `res.points_2d` float32 `(M, 2)`,
`res.points_3d` float32 `(M, 3)` (planar, z = 0) and `res.points_level`, the
level of each point.

`estimate_pose(..., full=True)` returns a `Pose` with the method used, the
number of points and inliers, the 6×6 covariance `cov` of `(rvec, tvec)` and
the other planar solution (`rvec_alt`, `tvec_alt`, `reproj_err_alt`): when
`reproj_err_alt` is close to `reproj_err` the tilt is ambiguous.

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
p.close_pass = True          # retry empty frames with a large threshold block (very close markers)
det = nf.FractalDetector("FRACTAL_5L_6", params=p)
det.params.adaptive_c = 5.0  # changes apply to the next detect()
```

### Video

```python
frames = [np.roll(image, 20 * k, axis=1) for k in range(5)]   # stand-in for a camera
tracker = nf.FractalTracker(det, auto_scale=True, use_prior=True)   # margin=0.25, pad=16
for frame in frames:
    res = tracker.detect(frame)           # det.detect(frame), plus levels read through the last result
    print(res.cell_px, res.contrast, res.bit_errors)   # reading quality per level

# on a drone's approach: also pick up the marker before it is first read
tracker = nf.FractalTracker(det, use_prior=True, capture_frames=3)
```

### More

- `detect(image, roi=(x, y, w, h))` — detect in a zero-copy sub-rectangle;
  corners come back in full-image coordinates.
- `detect_batch(images, num_threads=0, roi=None)` — parallel over frames.
- `detect(image, all_instances=True)` — every copy of the marker in view;
  `res.instance(k)` is copy `k`, ready for `estimate_pose`.
- `res.inconsistent_ids` — levels dropped because they do not fit the
  larger levels (a marker-like pattern elsewhere in the frame).
- `detect(image, prior=last_result)` — a level the plain reading misses by a
  few cells where it was a moment ago is kept and marked by `bit_errors > 0`
  (outer levels only; flight video: 10 more frames of 203).
- `FractalTracker(det, capture_frames=3)` — a quad within the same
  tolerance of a level's code that stays in place for 2 frames is read with
  it as prior in the 3rd (flight videos: the marker read 8–39 s earlier on
  the approach; no candidate on 2400 frames without it).
- `res.cell_px`, `res.contrast`, `res.bit_errors` — reading quality of every
  level, to weigh a detection instead of a yes/no.
- `diagnose(image)` — every candidate quad with the reason it was kept or
  dropped (`"duplicate"`, `"border"`, `"bits"`, `"inconsistent"`), its
  closest code and cell contrast: why a marker is not found.
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
  buffer together with 8-neighbour occupancy masks. Horizontal sums come from
  a SIMD prefix scan (AVX2 / NEON). Row stripes run in parallel.
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
  traffic. A quad the reference reader rejects is read again with each cell
  sampled exactly at its centre (the reference sampler is up to a pixel off,
  a whole cell for far markers).
- **Inner points.** FAST runs only around the marker; keypoint filtering and
  matching use grids instead of O(n²) scans.
- **Tracking** (`FractalTracker`): region-of-interest search around the
  previous detection, after Romero-Ramirez et al., *Speeded up detection of
  squared fiducial markers* (2018). Their other lever — shrinking the image to
  the previous marker size — was tested and rejected: it raised the corner
  error from 0.39 to 1.3 px.
- **Zero-copy input, GIL released**, a detector pool for `detect_batch`.

Intentional differences from the reference detector: the second bit reading
finds more levels (far, blurred, steeply tilted markers) — levels the
reference finds are unchanged; a frame where nothing is found is thresholded
again with a large block (`close_pass`, markers seen from very close); levels
that do not fit the larger levels are dropped (`inconsistent_ids`); with
`with_inner_points=True`, the FAST response threshold is computed over the
marker's neighbourhood instead of the whole frame. Reading with a `prior` is
off unless asked for.

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
