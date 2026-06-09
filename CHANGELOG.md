# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- `ArucoDetector` (ArUco Nano v6): `detect`, `estimate_pose`, `detect_batch`,
  `draw`.
- `FractalDetector` (fractal markers): `detect` / `detect_full` with inner-point
  correspondences, `detect_batch`, occlusion-robust `estimate_pose`
  (inner-points with outer-corner fallback), and `draw` for corners/ids and pose
  axes.
- GIL released during detection for multi-core batch scaling.
- Portable `manylinux` wheels with a minimal static OpenCV linked in.

[0.1.1]: https://github.com/Onikore/nanofractal/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Onikore/nanofractal/releases/tag/v0.1.0
