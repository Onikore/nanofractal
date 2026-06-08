# nanofractal

High-performance fiducial-marker detection for Python. `nanofractal` wraps two
compact, header-only C++ detectors with [nanobind](https://github.com/wjakob/nanobind):

- **ArUco Nano v6** — square markers (`ARUCO_MIP_36h12` and AprilTag `36h11`).
- **Fractal markers** — nested markers that stay detectable under heavy occlusion
  and expose many inner corner correspondences for accurate, long-range pose.

It is built for speed: **zero-copy** NumPy ↔ `cv::Mat`, the **GIL is released**
during detection, and a **parallel batch** API scales across cores.

```text
single-frame detect():   ~0.43 ms @ 640x480   ~1.3 ms @ 1280x720   ~2.9 ms @ 1920x1080
batch detect_batch():    ~3.2x throughput on 4 threads
```

> Measured on a desktop CPU with `max_attempts=1`; your numbers will vary.

---

## Installation

```bash
pip install nanofractal
```

Wheels bundle a minimal OpenCV, so no system OpenCV is required at runtime.

### Build from source

You need a C++17 compiler, CMake ≥ 3.18 and a development OpenCV
(`core`, `imgproc`, `calib3d`, `features2d`):

```bash
# Debian/Ubuntu
sudo apt-get install -y build-essential cmake libopencv-dev

pip install .
```

---

## Quick start

Inputs are plain NumPy `uint8` arrays — either `(H, W)` grayscale or `(H, W, 3)`
BGR, and **C-contiguous** (use `np.ascontiguousarray` if unsure). Any image loader
works; the examples use OpenCV.

### Detect ArUco / AprilTag markers

```python
import cv2
import nanofractal as nf

image = cv2.imread("scene.png")                  # (H, W, 3) uint8 BGR
det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12)  # or nf.Dict.APRILTAG_36h11

res = det.detect(image)
print(res.ids)        # int32   (N,)       e.g. [ 7 42]
print(res.corners)    # float32 (N, 4, 2)  clockwise corners, subpixel
```

### Estimate pose

`estimate_pose` runs `solvePnP` (IPPE) for every detected marker at once.

```python
import numpy as np

camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0,   0,   1]], dtype=np.float64)
dist_coeffs = np.zeros(5, dtype=np.float64)

rvecs, tvecs = det.estimate_pose(res.corners, camera_matrix, dist_coeffs,
                                 marker_size=0.05)   # marker side in metres
# rvecs, tvecs: float64 (N, 3) — rotation (Rodrigues) and translation per marker
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

### `ArucoDetector(dictionary=Dict.ARUCO_MIP_36h12, max_attempts=1)`
- `dictionary: Dict` — `ARUCO_MIP_36h12` or `APRILTAG_36h11`.
- `max_attempts: int` — retries per candidate with small corner jitter. `1` is
  fastest (real-time default); raise (up to ~10) for harder images.
- `detect(image) -> DetectionResult`
- `detect_batch(images, num_threads=0) -> list[DetectionResult]`
- `estimate_pose(corners, camera_matrix, dist_coeffs, marker_size) -> (rvecs, tvecs)`
  — `corners` is `(N, 4, 2)` float32; outputs are `(N, 3)` float64.

### `FractalDetector(config, marker_size=-1.0)`
- `config: str` — one of `FRACTAL_2L_6`, `FRACTAL_3L_6`, `FRACTAL_4L_6`,
  `FRACTAL_5L_6`.
- `marker_size: float` — outer marker side in metres; if set, `points_3d` is
  returned in metres (otherwise normalized).
- `detect(image, with_inner_points=False) -> DetectionResult`
- `detect_batch(images, num_threads=0) -> list[DetectionResult]`
- `estimate_pose(result, camera_matrix, dist_coeffs) -> (rvec, tvec, reproj_err) | None`
  — single-marker pose; uses inner+outer points when ≥ 4, else the 4 outer
  corners; `rvec`/`tvec` are float64 `(3,)`, `reproj_err` is RMS pixels.
- `draw(image, result, camera_matrix=None, dist_coeffs=None, rvec=None, tvec=None, axis_length=None) -> image`
  — draw outlines + ids (and frame axes when a pose is given) in place; `image`
  must be a writable BGR `uint8` array.

### `DetectionResult`
| field | dtype / shape | meaning |
|-------|---------------|---------|
| `ids` | int32 `(N,)` | marker ids |
| `corners` | float32 `(N, 4, 2)` | outer corners (subpixel, clockwise) |
| `points_2d` | float32 `(M, 2)` or `None` | inner+outer image points (fractal, `with_inner_points=True`) |
| `points_3d` | float32 `(M, 3)` or `None` | matching object points |

Empty results are returned as correctly-shaped empty arrays (`(0,)`, `(0, 4, 2)`),
never `None`.

### Errors
- Wrong dtype / non-contiguous input → `TypeError`.
- Unsupported shape, empty frame, invalid dictionary or fractal config → `ValueError`.

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
