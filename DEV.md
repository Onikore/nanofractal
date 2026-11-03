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

Three workflows. `ci.yml` is one run per push or pull request (one
notification), with these jobs:

| job | what |
|---|---|
| `lint` | `pre-commit` (ruff, ruff-format, clang-format, …); the other jobs wait for it |
| `test` | build and tests against the system OpenCV 4 (apt), coverage |
| `test-opencv5` | build and tests against a static OpenCV 5 (AVX2) |
| `types` | stubs match the extension (`mypy.stubtest`) |
| `asan` | tests under AddressSanitizer / UBSan |
| `bench` | speed gate: the base commit against the new one, see below |
| `docs`, `pages` | documentation site from `guide/`; published from `main` |

A newer push to the same branch cancels the run in progress; commits that
only touch notes (CHANGELOG, ROADMAP, the test plan, DEV, …) do not run it —
README and `guide/` do, their examples are tests. All jobs but `test` build
against the static OpenCV 5 cached by key (hash of `ci/build-opencv.sh`),
restored in about a second; installing `libopencv-dev` with apt takes from
30 s to over 7 min.

`docs` builds the documentation site from `guide/` with MkDocs Material
(`mkdocs.yml`, `--strict`: a broken link fails it) and `pages` publishes it
to <https://onikore.github.io/nanofractal/> on pushes to `main` once the
tests pass. Locally:

```bash
pip install mkdocs==1.6.1 mkdocs-material==9.7.7
mkdocs serve            # http://127.0.0.1:8000, reloads on edits
```

`bench-arm.yml` (manual) runs the tests on aarch64 and times NEON against the
scalar paths; `release.yml` builds and tests the wheels of every platform and
publishes on a `v*` tag.

To get mail only when something fails: GitHub → Settings → Notifications →
System → Actions → "Only notify for failed workflows".

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

The `bench` job of `ci.yml` does this on every push and pull request: three alternating rounds of the base commit and the new one on
one core, compared by `fractal_bench.py gate` (fastest median per cell). It
fails when the geometric mean of the cells is more than x1.10 slower or one
cell more than x1.25; the same build against itself gives x0.99–1.01.
`workflow_dispatch` takes another base (a tag, say) to compare releases.

## Environment note (ROS machines)

If the shell sources a ROS workspace, its `PYTHONPATH` pulls in pytest
plugins that crash collection; run Python with `env -u PYTHONPATH`.
