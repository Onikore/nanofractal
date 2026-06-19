# nanofractal

High-performance fiducial-marker detection for Python. `nanofractal` wraps two
compact, header-only C++ detectors with [nanobind](https://github.com/wjakob/nanobind):

- **ArUco Nano v6** — square markers: all standard OpenCV ArUco dictionaries
  (4×4, 5×5, 6×6, 7×7) plus `ARUCO_MIP_36h12` and AprilTag `36h11`.
- **Fractal markers** — nested markers that stay detectable under heavy occlusion
  and expose many inner corner correspondences for accurate, long-range pose.

It is built for speed: **zero-copy** NumPy ↔ `cv::Mat`, the **GIL is released**
during detection, and a **parallel batch** API scales across cores.

```text
single-frame detect():   ~0.43 ms @ 640×480   ~1.0 ms @ 1280×720   ~3.1 ms @ 1920×1080
detection_scale=0.5:     ~4× faster on the threshold/contour stage (corners refined at full res)
batch detect_batch():    ~3.2× throughput on 4 threads
```

> Measured on a desktop CPU with `max_attempts=1`; your numbers will vary.

---

## Installation

```bash
pip install nanofractal
```

Wheels are available for x86_64 and aarch64 Linux (manylinux). They bundle a
minimal OpenCV, so no system OpenCV is required at runtime.

### Build from source

You need a C++17 compiler, CMake ≥ 3.18 and a development OpenCV
(`core`, `imgproc`, `calib3d`, `features2d`):

```bash
# Debian/Ubuntu
sudo apt-get install -y build-essential cmake libopencv-dev

pip install .
```

#### Local dev build with CPU tuning

```bash
# Enable -march=native + -ffast-math for maximum local performance:
NF_NATIVE=1 pip install -e . --no-build-isolation
```

---

## Quick start

Inputs are plain NumPy `uint8` arrays — either `(H, W)` grayscale, `(H, W, 1)`
grayscale, or `(H, W, 3)` BGR, and **C-contiguous** (use `np.ascontiguousarray`
if unsure). Any image loader works; the examples use OpenCV.

### Generate ArUco markers

```python
import nanofractal as nf

# Generate a single 4×4 marker (id=7, 200×200 pixels, grayscale uint8)
marker = nf.generate_aruco(marker_id=7, size_px=200,
                           dictionary=nf.Dict.DICT_4X4_50, border_bits=1)

# Generate the external level of a fractal marker
config = "FRACTAL_5L_6"
fractal = nf.generate_fractal(config, size_px=400)  # uint8 grayscale
```

### Detect ArUco markers

```python
import cv2
import nanofractal as nf

image = cv2.imread("scene.png")                  # (H, W, 3) uint8 BGR

# Standard 4×4 dictionary (50 unique markers)
det = nf.ArucoDetector(nf.Dict.DICT_4X4_50)

# Or the legacy / AprilTag dictionaries:
# det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12)
# det = nf.ArucoDetector(nf.Dict.APRILTAG_36h11)

res = det.detect(image)
print(len(res))        # number of detected markers
print(res.ids)         # int32   (N,)       e.g. [ 7 42]
print(res.corners)     # float32 (N, 4, 2)  clockwise corners, subpixel

# Iterate over results
for marker_id, corners in res:
    print(f"Marker {marker_id}: {corners}")
```

### Tune detection parameters

```python
params = nf.DetectorParams()
params.min_contour_size    = 30    # detect smaller markers (default: 50)
params.adaptive_block_size = 11    # adaptive threshold window (must be odd, ≥3)
params.adaptive_c          = 7.0   # threshold constant (default: 7)
params.approx_poly_rate    = 0.05  # polygon approx rate (default: 0.05)

det = nf.ArucoDetector(nf.Dict.DICT_5X5_100, params=params)

# Or change params after creation:
det.params.min_contour_size = 80
```

For high-resolution input with reasonably large markers, `detection_scale` is the
single biggest speed lever — the dominant cost (`adaptiveThreshold` + `findContours`)
is already SIMD-optimized inside OpenCV, so the win comes from feeding it fewer
pixels. Corners are still refined at full resolution, and it works for **both**
`ArucoDetector` and `FractalDetector`:

```python
params = nf.DetectorParams()
params.detection_scale = 0.5   # ~4x faster threshold/contour stage @1080p

det  = nf.ArucoDetector(nf.Dict.DICT_4X4_50, params=params)
fdet = nf.FractalDetector("FRACTAL_5L_6", params=params)
```

FractalDetector also supports all the same parameters plus two extras:

```python
fparams = nf.DetectorParams()
fparams.subpix_win_size  = 4    # corner sub-pixel half-window (0 = off)
fparams.kfilter_min_dist = 10.0 # min pixel distance between FAST keypoints
```

### Estimate pose

`estimate_pose` runs `solvePnP` (IPPE_SQUARE) for every detected marker at once.

```python
import numpy as np

camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0,   0,   1]], dtype=np.float64)
dist_coeffs = np.zeros(5, dtype=np.float64)

rvecs, tvecs = det.estimate_pose(res.corners, camera_matrix, dist_coeffs,
                                 marker_size=0.05)   # marker side in metres
# rvecs, tvecs: float64 (N, 3) — rotation (Rodrigues) and translation per marker

# With reprojection errors per marker
rvecs, tvecs, reproj_errs = det.estimate_pose(
    res.corners, camera_matrix, dist_coeffs, marker_size=0.05, return_reproj=True
)
# reproj_errs: float64 (N,) — RMS reprojection error in pixels per marker
```

### Fisheye distortion model

Both detectors support OpenCV's fisheye distortion model:

```python
# Fisheye intrinsics with exactly 4 distortion coefficients (k1, k2, k3, k4)
camera_matrix_fisheye = np.array([[500, 0, 320],
                                  [0, 500, 240],
                                  [0,   0,   1]], dtype=np.float64)
dist_coeffs_fisheye = np.array([0.1, 0.01, -0.001, 0.0005], dtype=np.float64)

rvecs, tvecs = det.estimate_pose(
    res.corners, camera_matrix_fisheye, dist_coeffs_fisheye,
    marker_size=0.05, fisheye=True
)
```

### Smooth pose over time

```python
smoother = nf.PoseSmoother(process_noise=1e-4, measurement_noise=1e-2)

# In your frame loop:
rvec, tvec = det.estimate_pose(res.corners, K, D, marker_size=0.05)[0]
rvec_smooth, tvec_smooth = smoother.update(rvec, tvec)
```

### Detect fractal markers

```python
fdet = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.85)  # size in metres (optional)

res = fdet.detect(image)
print(res.ids, res.corners.shape)   # outer 4 corners of each fractal marker
```

### Fractal pose + visualization (occlusion-robust)

`FractalDetector.estimate_pose` returns one marker pose `(rvec, tvec, reproj_err)`
or `None`. It uses every visible inner **and** outer corner correspondence when
available (accurate, robust to occlusion) and otherwise falls back to the four
outer corners — so you never call `solvePnP` yourself or worry about the
empty-inner-points case. `reproj_err` (RMS pixels) lets you gate noisy poses.

```python
fdet = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.85)  # size in metres

res = fdet.detect(image, with_inner_points=True)
pose = fdet.estimate_pose(res, camera_matrix, dist_coeffs)
if pose is not None:
    rvec, tvec, reproj_err = pose      # rvec, tvec: float64 (3,); reproj_err: px
    fdet.draw(image, res, camera_matrix, dist_coeffs, rvec, tvec)  # corners + axes
```

`draw(image, result, ...)` overlays marker outlines, ids and (given a pose) the
frame axes in place — no `cv2.polylines`/`drawFrameAxes` boilerplate. Without a
pose, `fdet.draw(image, res)` just draws the outlines.

The raw correspondences are still exposed if you prefer to run PnP yourself:

```python
res.points_2d   # float32 (M, 2) image points  (None unless with_inner_points=True)
res.points_3d   # float32 (M, 3) object points (planar, z = 0)
```

### Draw ArUco markers with pose

```python
det = nf.ArucoDetector(nf.Dict.DICT_4X4_50)
res = det.detect(image)
rvecs, tvecs = det.estimate_pose(res.corners, K, D, marker_size=0.05)

# Draw outlines + ids + per-marker axes
det.draw(image, res, K, D, rvecs, tvecs, marker_size=0.05, inplace=True)
cv2.imshow("result", image)
```

### Non-destructive drawing

Pass `inplace=False` to draw on a copy (preserves the original):

```python
result_image = det.draw(image, res, K, D, rvecs, tvecs, inplace=False)
# image remains unchanged; result_image contains the annotated version
```

### Introspect dictionaries

```python
# Grid size (including border cells)
grid_size = nf.dict_grid_size(nf.Dict.DICT_4X4_50)  # returns 6

# Number of markers in the dictionary
num_markers = nf.dict_num_markers(nf.Dict.DICT_4X4_50)  # returns 50
num_apriltag = nf.dict_num_markers(nf.Dict.APRILTAG_36h11)  # returns 587
```

### Benchmark

Run a quick throughput benchmark (stdlib + NumPy only):

```bash
python -m nanofractal.bench --resolution 1280x720 --detector aruco --frames 200
# Output: latency, FPS, library versions, CPU architecture
```

### Parallel batch (offline throughput)

Process many frames across a thread pool. The GIL is released, so it scales with
cores. `num_threads=0` uses all cores.

```python
frames = [cv2.imread(p) for p in paths]            # list of uint8 arrays
results = det.detect_batch(frames, num_threads=0)  # list[DetectionResult]
for r in results:
    print(r.ids)
```

---

## API

### `Dict` — marker dictionaries

| Name | Markers | Inner bits | Notes |
|------|---------|------------|-------|
| `DICT_4X4_50` … `DICT_4X4_1000` | 50–1000 | 4×4 | fewest bits, fastest matching |
| `DICT_5X5_50` … `DICT_5X5_1000` | 50–1000 | 5×5 | |
| `DICT_6X6_50` … `DICT_6X6_1000` | 50–1000 | 6×6 | |
| `DICT_7X7_50` … `DICT_7X7_1000` | 50–1000 | 7×7 | most bits, best error detection |
| `ARUCO_MIP_36h12` | 250 | 6×6 | legacy ArUco MIP dictionary |
| `APRILTAG_36h11` | 587 | 6×6 | AprilTag 36h11 |

All dictionaries are identical to their OpenCV counterparts — markers printed
with `cv2.aruco.generateImageMarker` are detected directly.

### `ArucoDetector(dictionary=Dict.ARUCO_MIP_36h12, max_attempts=1, params=None)`
- `dictionary: Dict` — any `Dict` enum value.
- `max_attempts: int` — retries per candidate with small corner jitter. `1` is
  fastest (real-time default); raise (up to ~10) for harder images.
- `params: DetectorParams | None` — tuning parameters (see below). `None` uses defaults.
- `.params` — read/write access to the `DetectorParams` after creation.
- `detect(image) -> DetectionResult`
- `detect_batch(images, num_threads=0) -> list[DetectionResult]`
- `estimate_pose(corners, camera_matrix, dist_coeffs, marker_size, return_reproj=False, fisheye=False) -> (rvecs, tvecs) | (rvecs, tvecs, reproj_errs)`
  — `corners` is `(N, 4, 2)` float32. When `return_reproj=True` returns `(rvecs, tvecs, reproj_errs)`
  where reproj_errs is float64 `(N,)` per-marker RMS error. `fisheye=True` uses OpenCV fisheye model
  (dist_coeffs must be exactly 4).
- `draw(image, result, camera_matrix=None, dist_coeffs=None, rvecs=None, tvecs=None, marker_size=None, axis_length=None, inplace=True) -> image`
  — draw outlines + ids; with poses, also draw frame axes per marker. `inplace=True` (default) modifies and returns
  the input; `inplace=False` returns a copy (accepts read-only input).

### `FractalDetector(config, marker_size=-1.0, params=None)`
- `config: str` — one of `FRACTAL_2L_6`, `FRACTAL_3L_6`, `FRACTAL_4L_6`,
  `FRACTAL_5L_6`.
- `marker_size: float` — outer marker side in metres; if set, `points_3d` is
  returned in metres (otherwise normalized).
- `params: DetectorParams | None` — tuning parameters. `None` uses defaults.
- `.params` — read/write access to the `DetectorParams` after creation.
- `detect(image, with_inner_points=False) -> DetectionResult`
- `detect_batch(images, num_threads=0) -> list[DetectionResult]`
- `estimate_pose(result, camera_matrix, dist_coeffs, fisheye=False) -> (rvec, tvec, reproj_err) | None`
  — single-marker pose; uses inner+outer points when ≥ 4, else the 4 outer
  corners; `rvec`/`tvec` are float64 `(3,)`, `reproj_err` is RMS pixels. `fisheye=True`
  uses OpenCV fisheye model (dist_coeffs must be exactly 4).
- `draw(image, result, camera_matrix=None, dist_coeffs=None, rvec=None, tvec=None, axis_length=None, inplace=True) -> image`
  — draw outlines + ids (and frame axes when a pose is given). `inplace=True` (default) modifies
  and returns the input; `inplace=False` returns a copy.

### `DetectorParams`

Shared by both detectors. All fields are optional — defaults reproduce the
original hard-coded behaviour so existing code needs no changes.

| Field | Default | Description |
|-------|---------|-------------|
| `min_contour_size` | `-1` (auto) | Minimum contour perimeter in pixels. ArUco default: 50, Fractal: 120. |
| `adaptive_block_size` | `-1` (auto) | Adaptive threshold block size (odd, ≥ 3). ArUco default: 13; Fractal: scales with image width. |
| `adaptive_c` | `7.0` | Constant subtracted from the local threshold mean. |
| `approx_poly_rate` | `0.05` | Polygon approximation: `epsilon = perimeter × rate`. |
| `subpix_win_size` | `-1` (auto=4) | Corner sub-pixel half-window **(Fractal only)**; `0` to disable. |
| `kfilter_min_dist` | `10.0` | Minimum distance (px) between FAST keypoints **(Fractal only)**. |
| `detection_scale` | `1.0` | Downscale factor for the detection stage **(both detectors)**. `0.5` runs threshold/contour/decode on ¼ the pixels (≈ 4× faster); corners are mapped back and sub-pixel refined at full resolution. `min_contour_size` stays in original-image pixels. |

### `DetectionResult`
| field | dtype / shape | meaning |
|-------|---------------|---------|
| `ids` | int32 `(N,)` | marker ids |
| `corners` | float32 `(N, 4, 2)` | outer corners (subpixel, clockwise) |
| `points_2d` | float32 `(M, 2)` or `None` | inner+outer image points (fractal, `with_inner_points=True`) |
| `points_3d` | float32 `(M, 3)` or `None` | matching object points |

**Ergonomics:**
- `len(result)` — number of markers.
- `bool(result)` — `True` if any markers detected.
- `for mid, corners in result:` — iterate over `(marker_id, corners_array)` pairs.
- `repr(result)` — concise summary including marker count and ids.

Empty results are returned as correctly-shaped empty arrays (`(0,)`, `(0, 4, 2)`),
never `None`.

### Module-level functions

| Function | Returns | Purpose |
|----------|---------|---------|
| `generate_aruco(marker_id, size_px=200, dictionary=Dict.DICT_4X4_50, border_bits=1)` | `uint8 (size_px, size_px)` | Generate an ArUco marker image. |
| `generate_fractal(config, size_px=400)` | `uint8 (size_px, size_px)` | Generate the external level of a fractal marker. |
| `dict_grid_size(d: Dict)` | `int` | Full grid size (including border) for a dictionary. |
| `dict_num_markers(d: Dict)` | `int` | Number of markers in a dictionary. |

**Note on `generate_fractal`:** Returns only the outermost marker level, which is
detectable by `FractalDetector`. It is **not** a full multi-level nested composite.

### `PoseSmoother`

Temporal smoothing of 6-DOF pose using Kalman filtering or exponential moving average.

```python
smoother = nf.PoseSmoother(process_noise=1e-4, measurement_noise=1e-2, mode="kalman")

# Per frame:
rvec, tvec = smoother.update(rvec_measured, tvec_measured)

# Reset state (e.g. on marker loss):
smoother.reset()
```

- `mode="kalman"` — Kalman filter (default).
- `mode="ema"` — Exponential moving average.
- Rvec is smoothed component-wise (suitable for small inter-frame rotation changes).

### Errors
- Wrong dtype / non-contiguous input → `TypeError`.
- Unsupported shape, empty frame, invalid dictionary or fractal config → `ValueError`.
- Invalid camera intrinsics or distortion → `ValueError` with clear message.

---

## Performance notes

- **Zero-copy input.** A contiguous `uint8` array is wrapped as a `cv::Mat` over
  the same buffer — no copy. Non-contiguous or wrong-dtype inputs raise instead of
  silently copying.
- **GIL released** during the native detection, so other Python threads keep
  running and `detect_batch` scales.
- **Thread safety.** The ArUco detector is stateless and shared across batch
  workers. The fractal detector is not thread-safe, so `detect_batch` uses a pool
  of independent detectors (one per worker). A single detector object is fine to
  call from one thread at a time.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full version history.

**Latest features:**
- Marker generation (`generate_aruco`, `generate_fractal`).
- `ArucoDetector.draw()` with pose visualization.
- Per-marker reprojection errors and fisheye model support.
- `PoseSmoother` for temporal pose smoothing.
- Dictionary introspection (`dict_grid_size`, `dict_num_markers`).
- Benchmark CLI.
- aarch64 wheels.

---

## Citation

If you use this in research, please cite the original work:

- F. J. Romero-Ramirez, R. Muñoz-Salinas, R. Medina-Carnicer,
  *"Speeded up detection of squared fiducial markers"*, Image and Vision
  Computing, 76, 2018.
- S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, R. Medina-Carnicer,
  *"Generation of fiducial marker dictionaries using mixed integer linear
  programming"*, Pattern Recognition, 51, 2016.
- F. J. Romero-Ramirez, R. Muñoz-Salinas, R. Medina-Carnicer,
  *"Fractal Markers: A New Approach for Long-Range Marker Pose Estimation Under
  Occlusion"*, IEEE Access, 7, 2019.

## License

Apache-2.0. The vendored detectors (ArUco Nano, Fractal markers) are © their
authors and used under their terms; see `third_party/` and `PATCHES.md`.
