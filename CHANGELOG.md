# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Far markers: a quad the reference bit reader rejects gets a second
  reading — each cell sampled exactly at its centre (the reference sampler
  is up to a pixel off, a whole cell at 2–4 px per cell), on corners refined
  to sub-pixel when the first try fails. Synthetic far frames, outer level of
  `FRACTAL_4L_6`: found at 2.6 px/cell 0 → 80 %, 3.0 px/cell 0 → 100 %,
  3.5 px/cell 27 → 100 %; no detection on 300 frames of marker-like
  negatives; speed unchanged. Levels the reference finds keep exactly their
  corners; more inner levels are found (e.g. blur σ=2, 45–60° tilt, small
  markers), so `ids` and inner points can grow (R16).

### Added
- `FractalTracker(..., auto_scale=True, scale_target_px=400)`: also shrinks
  the searched region from the previous marker size (down to 0.35); corners
  are refined on the full frame and stay as accurate. 4K synthetic video:
  ×5.7 vs `detect` on one core (plain tracker ×2.5), ×3.9 on all cores (R6).
- `DetectionResult.points_level`: the level id of every inner point
  (`detect(..., with_inner_points=True)`), int32 `(M,)` (R19).
- Level consistency check: the homography of a larger level predicts the
  corners of each smaller one; a level found elsewhere (more than half its
  side away) is dropped and listed in `DetectionResult.inconsistent_ids`
  (also in `detect_batch` and `diagnose`, reason `"inconsistent"`). No real
  level dropped on 696 synthetic frames of all configurations, including
  strong uncorrected lens distortion (R18).
- `FractalDetector.diagnose(image, roi=None)`: the detection plus every
  candidate quad with the reason it was kept or dropped (`"accepted"`,
  `"duplicate"`, `"border"`, `"bits"`), its closest code, the number of wrong
  border cells and bits, and the cell contrast — to tell a reading failure
  (one bit off) from a missing contour (R16). Same detection as `detect`.
- Wheels for macOS (arm64 and x86_64, macOS 11+) and for CPython 3.14 (R1,
  R11a). The bundled OpenCV uses the pthreads backend on macOS as on Linux
  (its default there, Grand Central Dispatch, ignores `set_num_threads`).
- `DetectionResult.multiple_instances`: `True` when one marker level is seen
  in two separate places (two copies of the marker in view); detection still
  returns the largest copy of each level (R8a).
- `PoseSmoother.update(..., t=seconds)`: real time steps and a
  constant-velocity model (no lag on steady motion, dropped frames handled);
  `reproj_err` / `n_points` scale the measurement noise; `gate` rejects
  outliers by Mahalanobis distance (`last_rejected`, restart after
  `max_rejects`) (R13).
- `estimate_pose(..., full=True)` returns a `Pose` named tuple: `rvec`,
  `tvec`, `reproj_err`, `method` (`"inner"` / `"corners"`), `n_points`,
  `n_inliers` and `reproj_err_rel` (error / side of the largest detected
  level in px) (R2, R14).

### Tooling / CI
- aarch64 wheels build on GitHub's native arm64 runner instead of QEMU and run
  the full test-suite for every Python version.
- New CI job: static OpenCV 5.0 build with the AVX2 code paths (R11a).

### Documentation
- Threads under load: next to capture/encoding fewer `detect()` threads can
  be faster (Raspberry Pi 5 numbers) — `guide/05`, README (R15).

### Performance
- Adaptive threshold: horizontal box sums from a SIMD prefix scan (AVX2 /
  NEON; exact in 16-bit arithmetic), results packed straight to int8 in the
  large-window path, and the neighbour masks computed inside each row stripe
  while its rows are in cache. One core: `detect` ×1.17–1.34 at 4K,
  ×1.05–1.12 at 1080p; bit-identical to `cv::adaptiveThreshold` (new direct
  test, k = 3 … 255) (R7).
- aarch64: NEON version of the contour-start scan (it had an AVX2 path
  only). GitHub ARM runner, one core: `detect` 1.4–1.85× faster than the
  scalar build, results bit-identical (R4).
- `detection_scale` with a factor other than 0.5 / 0.25 (e.g. 0.75, 0.6,
  0.35) was slower than no scaling: OpenCV's generic `INTER_AREA` path cost
  ~3 ms single-threaded at 1080p. A separable area resize with the same
  weights (AVX2 in the x86-64 wheels, parallel row stripes) makes these
  factors 2.1–2.8× faster; detections are unchanged (bit-identical on the
  benchmark and robustness scenes; the resized image matches `cv::resize`
  exactly at 0.75 / 0.6 and within one grey level otherwise) (R3).

### Changed
- `estimate_pose` with 8 or more correspondences drops points off by more
  than `max(1.5 px, 3 × median)` once and solves again (if ≥ 8 points and
  ≥ 60 % remain); `reproj_err` is then the RMS over the kept points (R14).
  On noisy synthetic scenes the rotation error fell up to 5× and never grew
  beyond noise.
- `PoseSmoother` filters the orientation as a quaternion (tangent-space
  innovation) and keeps the output `rvec` continuous. Near a rotation of π
  (camera looking down at the marker) the component-wise filter averaged
  `rvec` values of opposite sign into garbage. Without the new arguments the
  results differ from 0.6.1 only by rounding-level amounts (< 1e-3 rad).

### Fixed
- `estimate_pose` without inner points treated the first detected level as
  the external marker: when only an inner level was visible, the distance was
  wrong by the ratio of the level sizes (×7 for the innermost level of
  `FRACTAL_3L_6`). It now uses the outer corners of every detected level at
  their real coordinates from the marker configuration (R2).
- `estimate_pose` on a marker facing the camera exactly: OpenCV's IPPE
  returned a wrong rotation (reprojection error of hundreds of pixels). A
  first solution worse than 1 px is now re-solved with `SOLVEPNP_SQPNP` and
  the better one kept.

## [0.6.1] - 2026-10-27

### Added
- `guide/`: a user guide (markers and printing, detection, pose, video,
  performance, tuning, API reference, troubleshooting) whose examples are run
  by `tests/test_guide.py`.

### Fixed
- Bounded reads in the reference detector's `getSubpixelValue` (ASan).
- `FractalTracker` typing (mypy / stubtest).

## [0.6.0] - 2026-10-27

### Removed (breaking)
- The ArUco detector and everything around it: `ArucoDetector`, `Dict`,
  `generate_aruco`, `dict_grid_size`, `dict_num_markers`, `to_opencv`,
  `from_opencv`, and `benchmarks/compare_opencv.py`. nanofractal is now a
  fractal-marker library.

### Added
- Optimised detection pipeline (`src/fractal_fast.hpp`): fused adaptive
  threshold, SIMD-assisted and band-parallel contour tracing, exact
  `approxPolyDP` replicas (OpenCV 4 and 5), parallel decoding. Outer corners are
  bit-identical to the reference detector. With the wheels' OpenCV 4.10:
  3.2–4.6× faster on one core and 6–19× with all cores at 1080p / 4K;
  `detect_batch` ~3.6× more frames/s.
- `set_num_threads(n)` / `get_num_threads()` — threads per `detect()` call.
- `FractalTracker` — video helper that searches near the previous detection
  (after Romero-Ramirez et al. 2018) with full-frame fallback; same results,
  ~2.2× faster at 4K.
- `generate_fractal(..., nested=True)` renders the full nested marker
  (previously only the external level).
- OpenCV 5 support (OpenCV 4 keeps working).
- `benchmarks/fractal_bench.py` (latency matrix, dump/compare of detections),
  deterministic benchmark scenes in `nanofractal.bench`, and
  `tests/test_reference_equivalence.py`.

### Changed
- `with_inner_points=True` runs FAST only around the marker; the keypoint
  response threshold is now taken from the marker's neighbourhood.
- `robustness.py` measures per-level detection, inner points and pose error.
- Builds no longer use `-ffast-math`.
- Windows wheels are built with AVX2 (`/arch:AVX2`), like the x86_64 Linux
  wheels (`-march=x86-64-v3`).

### Fixed
- Level 3 of `FRACTAL_5L_6` was never detected in `-ffast-math` builds (the CI
  wheels) because the submarker mask extent was truncated instead of rounded.
- `DetectorParams` changed after constructing a `FractalDetector` were
  validated but never used by detection.

## [0.5.0] - 2026-06-20

### Added
- **Windows wheels** (`win_amd64`, CPython 3.9–3.13) — built with the prebuilt
  OpenCV Windows release and `delvewheel`-bundled, alongside the existing Linux
  x86_64/aarch64 wheels.

### Changed
- **Releases publish via PyPI OIDC Trusted Publishing** — no stored API token.
- Added a `LICENSE` file (Apache-2.0) and full PyPI metadata (classifiers,
  authors, project URLs).
- Pinned a SHA-256 checksum on the downloaded OpenCV source tarball (CI build).
- README performance section reworked to point at the real benchmarks
  (`python -m nanofractal.bench`, `benchmarks/`) instead of stale hard-coded
  figures. More tests (`detection_scale`, ROI boundaries, bench smoke).

## [0.4.0] - 2026-06-19

### Added
- **ROI detection**: `detect(image, roi=(x, y, w, h))` (and `detect_batch`) on both
  detectors — runs on a zero-copy sub-rectangle and returns corners in full-image
  coordinates.
- **`refine_corners(image, corners, win_size=5)`**: standalone subpixel corner
  refinement (`cornerSubPix`).
- **`to_opencv` / `from_opencv`**: converters between `DetectionResult` and the
  `cv2.aruco` `(corners, ids)` format.
- **`examples/`**: runnable scripts (`detect_image`, `pose_estimation`,
  `batch_processing`, `webcam`).
- **`benchmarks/`**: `compare_opencv.py` (head-to-head vs `cv2.aruco`) and
  `robustness.py` (detection rate + pose error under blur / rotation / perspective
  / noise / scale, synthetic ground truth) with a `real_images/` slot. `cv2` lives
  in the optional `[bench]` extra.

### Tooling / CI
- **Coverage**: `pytest-cov` in CI (`--cov`) + Codecov badge.
- **Memory safety**: optional `-DNF_SANITIZE=ON` build (ASan + UBSan) and an `asan`
  CI job running the suite under sanitizers; plus 4K/8K and repeated-call tests.
- **Faster releases**: static OpenCV is cached across releases (`actions/cache`),
  the wheel build runs as a per-arch parallel matrix (x86_64 / aarch64), and
  aarch64 wheels are tested on a single CPython under QEMU — cutting the previous
  ~1 h aarch64 release time on cache hits.
- Type stubs extended (`roi`, `refine_corners`, `to_opencv`, `from_opencv`);
  `stubtest` stays green.

## [0.3.0] - 2026-06-19

### Added
- **Marker generation**: `generate_aruco(marker_id, size_px, dictionary, border_bits)` and `generate_fractal(config, size_px)` for creating marker images.
- **`ArucoDetector.draw()`**: draw marker outlines, ids, and pose axes; mirrors FractalDetector's interface.
- **`inplace` parameter**: both detectors' `draw()` methods now support `inplace=True` (default; modifies input) and `inplace=False` (returns a copy, accepts read-only input).
- **Per-marker reprojection errors**: `ArucoDetector.estimate_pose(..., return_reproj=True)` returns `(rvecs, tvecs, reproj_errs)`.
- **Fisheye distortion model**: both detectors' `estimate_pose()` methods accept `fisheye=True` (requires exactly 4 distortion coefficients).
- **`PoseSmoother`**: temporal 6-DOF pose smoothing via Kalman filter or exponential moving average.
- **Dictionary introspection**: `dict_grid_size(d)` and `dict_num_markers(d)` for querying dictionary properties.
- **`DetectionResult` ergonomics**: `len(result)`, `bool(result)`, iteration over `(id, corners)` pairs, and concise `repr()`.
- **`(H,W,1)` grayscale input**: in addition to `(H,W)` and `(H,W,3)`.
- **Detector `__repr__`**: ArucoDetector and FractalDetector now have readable string representations.
- **Benchmark CLI**: `python -m nanofractal.bench` for throughput benchmarking (stdlib + NumPy only).
- **Type stubs**: full `.pyi` stubs ship; CI runs mypy stubtest to keep them in sync.
- **aarch64 wheels**: Linux x86_64 and aarch64 (ARM64) wheels built via QEMU.

### Changed
- **Pose solver**: both detectors now use `solvePnP` with `SOLVEPNP_IPPE_SQUARE` for more robust estimation.
- **Camera intrinsics validation**: explicit shape and finiteness checks with clear error messages.
- **`DetectorParams` validation**: adaptive_block_size, min_contour_size, and other fields validated at detect-time.

## [0.2.0] - 2026-06-16

### Added
- `detection_scale` parameter: opt-in downscale of the threshold/contour/decode
  stage for both detectors (~4× faster at 1080p; corners refined at full resolution).
  New `DetectorParams` field.

### Changed
- Lower-overhead decode: ArUco dictionary code tables are cached per-detector
  instead of rebuilt every frame; marker-id matching and rotation now run on
  stack buffers with no per-candidate heap allocation (both detectors).

### Performance
- Pinned SIMD baseline: CI wheels build OpenCV with an explicit `SSE4_2` baseline
  and `AVX`/`AVX2`/`AVX512` runtime dispatch.

## [0.1.1] - 2026-06-09

### Fixed
- Suppressed nanobind's `leaked instances` warning printed at interpreter/kernel
  shutdown. It fired whenever user code still held a detector reference at
  teardown (e.g. a `cv2.VideoCapture` worker thread left running after `Ctrl+C`,
  or an interactive shell pinning the frame via `sys.last_traceback`). The
  warning was purely cosmetic — the bindings hold no Python references and
  nothing leaks during operation; the OS reclaims the memory at exit. Disabled
  via `nb::set_leak_warnings(false)` in the module initializer so end users no
  longer have to add their own cleanup to silence it.

## [0.1.0] - 2026-06-08

### Added
- Initial release: high-performance fiducial marker detection.
- `ArucoDetector` (ArUco Nano v6): `detect`, `estimate_pose`, `detect_batch`.
- `FractalDetector` (fractal markers): `detect` / `detect_full` with inner-point
  correspondences, `detect_batch`, occlusion-robust `estimate_pose`
  (inner-points with outer-corner fallback), and `draw` for corners/ids and pose
  axes.
- GIL released during detection for multi-core batch scaling.
- Portable `manylinux` wheels with a minimal static OpenCV linked in.

[0.5.0]: https://github.com/Onikore/nanofractal/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Onikore/nanofractal/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Onikore/nanofractal/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Onikore/nanofractal/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Onikore/nanofractal/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Onikore/nanofractal/releases/tag/v0.1.0
