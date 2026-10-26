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
`ci/build-opencv.sh`, which is what the Linux wheels use).

## Tests

```bash
python -m pytest tests -q --ignore tests/test_benchmarks.py
```

`tests/test_reference_equivalence.py` checks the optimised pipeline
(`src/fractal_fast.hpp`) against the upstream reference detector
(`third_party/nanofractal.h`, exposed as `_reference_detect`) and the
band-parallel contour tracer against sequential `cv::findContours`. Any
change to the hot path must keep those exact.

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
(`taskset -c N` on Linux); all-core numbers with `--threads 0`.

## Environment note (ROS machines)

If the shell sources a ROS workspace, its `PYTHONPATH` pulls in pytest
plugins that crash collection; run Python with `env -u PYTHONPATH`.
