# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.5] - 2026-11-03

### Changed
- Reading with a size hint (`expected_side_px`) accepts a cell contrast of
  30 grey levels instead of 40. Flight videos (8027 frames): 4780 -> 4818
  frames with the marker, the first read of one approach 14 s earlier (frame
  161 -> 91, an overexposed print), lookalikes on 92 frames instead of 53;
  where there is no marker (921 frames x 5 windows) still no detection and no
  lookalike. A floor of 25 adds no frame. Plain `detect`, `prior` and capture
  are unchanged.
- `fractal_fast.hpp`: the clockwise corner order, the 90-degree turn of a
  code grid, the code comparison and the two grid readings shared by the
  tolerant reads and `diagnose()` are one helper each instead of copies;
  `check_levels` uses `predicts`. Every reading path (plain, prior, capture,
  size hint with lookalikes, `diagnose`) gives identical results on 810
  flight frames.
- `FractalTracker.reset()` also forgets the age of the last result.
- `release.yml` creates the GitHub Release of each version tag after the
  PyPI upload: the CHANGELOG section of that version as notes, the wheels and
  the sdist attached.

### Documentation
- Documentation site on GitHub Pages: <https://onikore.github.io/nanofractal/>,
  built from `guide/` with MkDocs Material (`mkdocs.yml`; MkDocs 1.6.1 and
  Material 9.7.7 pinned) by the `docs` job of `ci.yml` (`--strict`) and
  published from `main` by the `pages` job once the tests pass. README,
  `DEV.md` and the package metadata (`Documentation` URL) point to it.
- `guide/09-landing.md`: a landing recipe — one step per frame (search,
  steer to a lookalike, pose), the height from a rangefinder, when to trust
  the pose (running median of `Pose.sd_t`), what cannot be fixed by the
  detector. Runnable, executed by the test-suite.
- `guide/06`: what the parameter sweeps on flight video showed (block,
  `adaptive_c`, `approx_poly_rate`: within 1 % of the defaults); the size
  hint lowers `min_contour_size` for its call.
- `estimate_pose(full=True)` and `Pose.cov` docstrings name the covariance,
  `sd_t` / `sd_rot_deg` and the second solution; `FractalTracker.detect` has
  a docstring.

## [0.7.4] - 2026-11-03

### Added
- `detect(..., expected_side_px=(lo, hi))` and
  `FractalTracker.detect(..., expected_side_px=...)`: the outer level's side
  in pixels is known (from the height). Quads of a level's expected size are
  read with up to 3 wrong border cells and `sz^2 / 10` wrong data cells (14
  of 144 for the outer level of `FRACTAL_3L_6`), the minimum contour drops to
  fit the expected size, and near misses (within `sz^2 / 7` cells) come back
  as `res.lookalikes` (`Lookalike`: corners, centre, level, border and data
  errors, contrast). Flight videos (8027 frames, window +-30 % around the
  marker's size, tracker with `capture_frames=3`): 4480 -> 4780 frames with
  the marker, first read 7-36 s earlier on five approaches, lookalikes on 53
  more frames; no detection and no lookalike on 611 frames without a marker
  with every window from 20 to 96 px (closest quad there 25+ cells off the
  code, the marker on the approach 9-24). Plain `detect` is unchanged
  (bit-identical). Cost with the hint: +0.19-0.33 ms per frame, one core.
  A wrong hint never loses a marker the plain reading finds (same corners;
  tested with windows of 5-6, 20-25 and 300-400 px); windows down to 5 px
  read nothing false on 421 frames without a marker, but a tiny window
  doubles the time of such a frame (minimum contour 16 px).
- `Pose.sd_tvec`, `Pose.sd_t`, `Pose.sd_rot_deg`: standard deviations from
  `cov`, `sd_t` as one number for the position. With 4 points `sd_t` varies
  about x5 from frame to frame (2 degrees of freedom), +-13 % with every
  level.

### Changed
- CI: `ci.yml`, `lint.yml`, `types.yml`, `asan.yml` and `bench.yml` are one
  workflow (`ci.yml`, jobs `lint`, `test`, `test-opencv5`, `types`, `asan`,
  `bench`): one run and one notification per push instead of five. Lint gates
  the other jobs, a newer push cancels the run in progress, and commits that
  only touch notes skip CI. `DEV.md` says how to get mail on failures only.

### Documentation
- Size hint and pose uncertainty: `guide/04` section with a runnable example
  (and what a wrong height does), `guide/03` (`sd_t`, its spread with 4
  points), `guide/05`, `guide/07`, `guide/08`, README. `ROADMAP.md`: R25,
  R26, and what is not taken from the same request (contrast normalisation).
- README: the Python examples are self-contained (a generated test frame
  instead of `scene.png`, a camera matrix, stand-in frames) and run by
  `tests/test_guide.py` with the guide's; the performance table re-measured
  on 0.7.3 (OpenCV 4.10, AVX2: 1080p `detect` 3.21 ms on one core, 0.99 ms
  on all; `detect_batch` ~1500 frames/s), noting the cost of `close_pass`
  on frames with no marker.
- `ROADMAP.md`: the upstream report on `getSubpixelValue` is not sent;
  capturing right after a track is lost was measured on the flight videos
  (no gain: 4480 frames either way) and is not taken.
- `ROADMAP.md`: threshold block, `adaptive_c` and upscaling measured on the
  flight videos (8027 frames, tracker with `capture_frames=3`): no block size
  beats the automatic one, `adaptive_c = 3` gives +0.8 % but loses frames
  under shadow on the synthetic set, upscaling x1.5 / x2 loses 90 / 123
  frames — all kept as they are.

## [0.7.3] - 2026-11-03

### Added
- `FractalTracker(..., capture_frames=N)` (0 = off): picks up a marker that is
  in view but not read yet. While there is no track, quads within the
  `prior` tolerance of a level's code (grids of 10+ cells) are candidates;
  one of the same level that stays in place (centre within a side, size
  within x1.5) for `N - 1` frames is read with it as `prior` in the next
  frame. Implies `use_prior`. Flight videos (autolanding-pi, 28.10, 8027
  frames): with `N = 3` the marker is first read 2-39 s earlier on four
  approaches (4292 -> 4480 frames with the marker); no candidate in 1242
  flight frames without a readable marker nor in 1200 synthetic frames of
  marker-like objects (all four configurations); 300 static scenes held for
  4 frames give no detection. Plain `detect` is bit-identical to 0.7.2. Cost
  while there is no track (cluttered frame, one core): +0.06 ms at 640x480,
  +0.13 ms at 1280x720, +0.77 ms at 1920x1080.
- `benchmarks/frames_report.py --capture N`.
- `.gitignore`: `videos/` (flight videos stay local).
- CI speed gate (`.github/workflows/bench.yml`): the base commit and the new
  one are built with the same OpenCV and timed on one core in three
  alternating rounds; a geometric-mean slowdown above x1.10 or one cell above
  x1.25 fails the run. `fractal_bench.py gate` does the comparison (fastest
  median per cell over the rounds); the same build against itself gives
  x0.994–1.003.
- `tests/test_reading_acceptance.py`: floors on the synthetic shadow-edge,
  occlusion and marker-like-negative frames of `benchmarks/reading.py`.

### Changed
- CI: the `bench`, `types` and `asan` jobs build against the cached static
  OpenCV 5 of `ci.yml` (restored in ~1 s) instead of `apt-get install
  libopencv-dev`, which took 27 s to over 7 min depending on the mirror. The
  system-OpenCV job of `ci.yml` keeps apt, since testing that OpenCV is its
  purpose.

### Documentation
- `DEV.md`: the CI workflows, the speed gate, the acceptance tests, timing
  two builds in alternating rounds. `ROADMAP.md` rules mention the gate.
  `examples/README.md`: plain `python` commands (the ROS note moved to a
  pointer to `DEV.md`) and a pointer to video tracking with `use_prior`.
- Capture: `guide/04` section with a runnable example, `guide/05` decision
  row, `guide/07` API, `guide/08` (found late on the approach; overexposed
  prints), README. `ROADMAP.md` records the R21 and R23 acceptance on the
  flight videos of 28.10 (R23: 3757 -> 4285 frames with the marker; R21: 52
  more frames at 0.12-0.5 m) and R24; `RPI_TEST_PLAN.md` closes R23 and adds
  R24 (`--capture 3`).

## [0.7.2] - 2026-11-02

### Added
- `detect(..., prior=last_result)` and `FractalTracker(..., use_prior=True)`:
  a quad no reading decodes, lying where a level of the last detection was
  (centre within one side, size within ×1.5), is compared with that level's
  code only and accepted with up to (n−2)²/18 wrong data cells and one wrong
  border cell — grids of 10+ cells only (8 of 144 for the outer level of
  `FRACTAL_3L_6`). On a landing flight video (autolanding-pi, 203 frames)
  3 more frames with the marker (5.6 px/cell, 4–8 wrong cells under blur)
  and 7 more with the outer level (one cell under a shadow edge); no
  detection on 600 frames of marker-like objects with a prior placed on
  every quad. Without a prior nothing changes (R23).
- `DetectionResult.cell_px`, `contrast`, `bit_errors`: the reading quality
  of every level — pixels per cell, mean bright minus mean dark cell at the
  final corners, cells read wrong (> 0 only through a prior) — to weigh a
  detection instead of a yes/no. `detect` speed unchanged (R22).

### Documentation
- `guide/04`: keeping a marker at the edge of range (`use_prior`, reading
  quality); `guide/07`: the new fields and arguments. ROADMAP: what a landing
  flight video showed (most losses are frames with no whole level in view)
  and which requests are not taken yet, with the reasons.
- Documentation brought up to date: README (0.7.1/0.7.2 features, the
  close pass among the differences from the reference), `guide/01` range
  re-measured (reliable from 3–3.5 px per cell, 40–50 px of side; at least
  one whole level must be in the frame), `guide/02` reading quality,
  `guide/05` decision table (`use_prior`, `close_pass`), `guide/07`
  (`prior`, `detect_batch` without quality fields, `Diagnosis.second_pass`),
  `guide/08` (marker cut by the frame edge, flicker, weighing a pose),
  `PUBLISHING.md`, `RPI_TEST_PLAN.md` (0.7.2, close range, 3–8 m).
- `benchmarks/frames_report.py`: reading-quality columns and `--prior` to
  read a folder of frames as a video.

## [0.7.1] - 2026-10-28

### Added
- Markers seen from very close: when a frame yields nothing, it is
  thresholded again shrunk to ~320 px wide with a block of a fifth of that
  width (`DetectorParams.close_pass`, on by default). The automatic block
  (11 px at 1280 wide) is far smaller than cells of 60 px and more, and on a
  dark frame their contours fall apart: real frames at 0.12 m (report from
  autolanding-pi) gave 0/20; synthetic dark close frames 0/60 → 60/60.
  Frames where the first pass finds a level are bit-identical to 0.7.0; no
  false detection on 300 frames of marker-like objects; a frame without a
  marker costs 0.25–0.5 ms more on one core, 0.1–0.15 ms on all cores
  (x86, 720p–1080p). `Diagnosis.second_pass` tells such detections apart
  (R21).

### Documentation
- Raspberry Pi 5 measurements: NEON is 1.5–2× faster than the scalar build
  on one core; thread scaling is limited by memory from about 960×720 up
  (`guide/05` table); `benchmarks/pi_threads.py` measures it on the board.

## [0.7.0] - 2026-10-28

### Added
- `Pose` (`estimate_pose(..., full=True)`) gains `cov`, the 6×6 covariance
  of `(rvec, tvec)` from the inlier residuals (matches Monte Carlo scatter
  within ±12 %, pinhole and fisheye), and the other planar solution
  `rvec_alt`, `tvec_alt`, `reproj_err_alt` to judge flip ambiguity. Only
  computed with `full=True` (~60 µs); the plain call is unchanged (R20).
- `detect(..., all_instances=True)`: every copy of the marker in view, its
  levels grouped by geometry (the largest level of a copy predicts its
  smaller ones); `DetectionResult.instances` / `points_instance`,
  `n_instances`, `instance(k)` for one copy (pose per copy). The default
  (largest of each level) is unchanged (R8).
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
- `benchmarks/frames_report.py`: a folder of real frames to CSV (levels,
  dropped levels, copies, the closest rejected quad, pose with covariance),
  used by `RPI_TEST_PLAN.md`.
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
- Far markers: a quad the reference bit reader rejects gets a second
  reading — each cell sampled exactly at its centre (the reference sampler
  is up to a pixel off, a whole cell at 2–4 px per cell), on corners refined
  to sub-pixel when the first try fails. Synthetic far frames, outer level of
  `FRACTAL_4L_6`: found at 2.6 px/cell 0 → 80 %, 3.0 px/cell 0 → 100 %,
  3.5 px/cell 27 → 100 %; no detection on 300 frames of marker-like
  negatives; speed unchanged. Levels the reference finds keep exactly their
  corners; more inner levels are found (e.g. blur σ=2, 45–60° tilt, small
  markers), so `ids` and inner points can grow (R16).
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
- `inconsistent_ids` could keep the previous frame's value on a frame with
  no candidate quad.
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

[Unreleased]: https://github.com/Onikore/nanofractal/compare/v0.7.2...HEAD
[0.7.2]: https://github.com/Onikore/nanofractal/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/Onikore/nanofractal/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/Onikore/nanofractal/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Onikore/nanofractal/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Onikore/nanofractal/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Onikore/nanofractal/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Onikore/nanofractal/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Onikore/nanofractal/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Onikore/nanofractal/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Onikore/nanofractal/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Onikore/nanofractal/releases/tag/v0.1.0
