# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
