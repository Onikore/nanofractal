# Development notes

## Build

The extension is built by scikit-build-core + CMake (nanobind) against
OpenCV 4.x (`core imgproc calib3d features2d`) or 5.x
(`core imgproc geometry features`). An editable install recompiles after C++
changes:

```bash
python -m pip install -e ".[test]"
NF_MARCH=x86-64-v3 python -m pip install -e ".[test]"   # AVX2, like the x86_64 wheels
```

`CMAKE_PREFIX_PATH` selects a non-system OpenCV (e.g. a static build from
`ci/build-opencv.sh`, which is what the Linux and macOS wheels use;
`OPENCV_VERSION=5.0.0 OPENCV_CACHE_DIR=<dir> bash ci/build-opencv.sh` builds
the OpenCV 5 used by the `test-opencv5` CI job).

## Tests

```bash
python -m pytest tests -q --ignore tests/test_benchmarks.py
```

`tests/test_reference_equivalence.py` checks the optimised pipeline
(`src/fractal_fast.hpp`) against the upstream reference detector
(`third_party/nanofractal.h`, exposed as `_reference_detect`) and the
band-parallel contour tracer against sequential `cv::findContours`. Any
change to the hot path must keep those exact.

`tests/test_reading_acceptance.py` keeps floors on the synthetic shadow-edge,
occlusion and marker-like-negative frames of `benchmarks/reading.py`;
`tests/test_far_reading.py` on far markers (~3 px per cell). A change that
loses frames there fails. `tests/test_guide.py` runs every example in
`guide/`. `tests/test_smoke.py::test_version` needs an installed package (it
fails when the extension is only on `PYTHONPATH`).

## CI

| workflow | what |
|---|---|
| `ci.yml` | build and tests against the system OpenCV 4 (apt) and a static OpenCV 5 (AVX2), coverage |
| `asan.yml` | tests under AddressSanitizer / UBSan |
| `types.yml` | stubs match the extension (`mypy.stubtest`) |
| `lint.yml` | `pre-commit` (ruff, ruff-format, …) |
| `bench.yml` | speed gate: the base commit against the new one, see below |
| `bench-arm.yml` | manual: aarch64 tests and NEON against scalar timings |
| `release.yml` | wheels for every platform, tested; publishes on a `v*` tag |

`asan`, `types` and `bench` build against the static OpenCV 5 that `ci.yml`
caches (key: hash of `ci/build-opencv.sh`), restored in about a second;
installing `libopencv-dev` with apt takes from 30 s to over 7 min.

## Performance work

```bash
python benchmarks/fractal_bench.py dump --out ref.npz     # before the change
python benchmarks/fractal_bench.py run --json base.json
# ... change, rebuild ...
python benchmarks/fractal_bench.py compare ref.npz        # identical detections?
python benchmarks/fractal_bench.py run --vs base.json     # speedups
python benchmarks/robustness.py --json after.json         # quality unchanged?
```

Measure single-core numbers with `--threads 1` and a pinned core
(`taskset -c N` on Linux); all-core numbers with `--threads 0`. Time the two
builds in alternating rounds, not one after the other: on a laptop the same
build drifts by up to 16 % between back-to-back runs.

`bench.yml` does this on every push and pull request that touches the C++
or the build: three alternating rounds of the base commit and the new one on
one core, compared by `fractal_bench.py gate` (fastest median per cell). It
fails when the geometric mean of the cells is more than x1.10 slower or one
cell more than x1.25; the same build against itself gives x0.99–1.01.
`workflow_dispatch` takes another base (a tag, say) to compare releases.

## Environment note (ROS machines)

If the shell sources a ROS workspace, its `PYTHONPATH` pulls in pytest
plugins that crash collection; run Python with `env -u PYTHONPATH`.
