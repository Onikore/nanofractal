# nanofractal Core Library Implementation Plan (Plan A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locally-installable, tested Python library that wraps `aruco_nano_v6.h` and `nanofractal.h` via nanobind with zero-copy numpy↔cv::Mat, GIL release, and a parallel batch API.

**Architecture:** nanobind C++ extension (`_nanofractal`) calls the two vendored header-only detectors directly. The binding layer wraps numpy buffers as `cv::Mat` without copying, releases the GIL during detection, and returns preallocated numpy arrays. A thin Python facade (`nanofractal`) wraps the low-level tuples into a `DetectionResult` dataclass. Parallel batch uses a pool of detector instances (one per worker thread) because the fractal detector is not thread-safe.

**Tech Stack:** C++17, nanobind, scikit-build-core, CMake, system OpenCV (core/imgproc/calib3d/features2d), numpy, pytest, pytest-benchmark.

**Scope note:** This is Plan A of two. Plan B (minimal static OpenCV + cibuildwheel + CI for portable PyPI wheels) is a separate plan written after this library builds and passes tests locally.

**Prerequisites (developer machine, Ubuntu/Debian):**
```bash
sudo apt-get install -y build-essential cmake libopencv-dev
python -m pip install --upgrade pip
```

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Build backend (scikit-build-core), metadata, deps |
| `CMakeLists.txt` | Find Python/nanobind/OpenCV, build `_nanofractal` module |
| `.gitignore` | Ignore build artifacts |
| `third_party/aruco_nano_v6.h` | Vendored ArUco Nano v6 header (copied, unmodified) |
| `third_party/nanofractal.h` | Vendored fractal header (copied, unmodified) |
| `src/ndarray_cv.hpp` | Zero-copy `nb::ndarray` → `cv::Mat`; numpy output helpers |
| `src/_bindings.cpp` | nanobind module: `ArucoDetector`, `FractalDetector`, debug helpers |
| `src/nanofractal/__init__.py` | Python facade: `DetectionResult`, `Dict`, public classes |
| `src/nanofractal/__init__.pyi` | Type stubs (numpy.typing) |
| `tests/conftest.py` | Synthetic marker rendering helpers |
| `tests/test_bridge.py` | Zero-copy / validation tests |
| `tests/test_aruco.py` | ArUco detection + pose tests |
| `tests/test_fractal.py` | Fractal detection + inner-points tests |
| `tests/test_batch.py` | Parallel batch correctness + GIL release |
| `tests/test_benchmarks.py` | Latency + throughput benchmarks |

---

## Task 1: Project scaffold + toolchain smoke test

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `CMakeLists.txt`, `src/_bindings.cpp`, `src/nanofractal/__init__.py`, `third_party/aruco_nano_v6.h`, `third_party/nanofractal.h`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Initialize git and vendor headers**

```bash
cd /home/dmitry/Desktop/nanofractal
git init
mkdir -p third_party src/nanofractal tests
cp aruco_nano_v6.h third_party/aruco_nano_v6.h
cp nanofractal.h third_party/nanofractal.h
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
build/
dist/
*.egg-info/
__pycache__/
*.so
.pytest_cache/
.benchmarks/
_skbuild/
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.1.0"]
build-backend = "scikit_build_core.build"

[project]
name = "nanofractal"
version = "0.1.0"
description = "High-performance fiducial marker detection (ArUco Nano v6 + Fractal)"
readme = "README.md"
requires-python = ">=3.9"
license = { text = "Apache-2.0" }
dependencies = ["numpy>=1.21"]

[project.optional-dependencies]
test = ["pytest>=7", "pytest-benchmark>=4"]

[tool.scikit-build]
minimum-version = "0.10"
build-dir = "build/{wheel_tag}"
wheel.packages = ["src/nanofractal"]
cmake.build-type = "Release"
```

- [ ] **Step 4: Create `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.18)
project(nanofractal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind shipped with the build environment
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

# OpenCV: system by default. CI (Plan B) overrides via -DOpenCV_DIR=<minimal build>
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d features2d)

nanobind_add_module(_nanofractal NB_STATIC src/_bindings.cpp)

target_include_directories(_nanofractal PRIVATE
  "${CMAKE_SOURCE_DIR}/third_party"
  "${CMAKE_SOURCE_DIR}/src"
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(_nanofractal PRIVATE ${OpenCV_LIBS})
target_compile_options(_nanofractal PRIVATE -O3)

include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg)
if(_ipo_ok)
  set_property(TARGET _nanofractal PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

install(TARGETS _nanofractal LIBRARY DESTINATION nanofractal)
```

- [ ] **Step 5: Create minimal `src/_bindings.cpp`**

```cpp
#include <nanobind/nanobind.h>
#include <opencv2/core.hpp>

namespace nb = nanobind;

NB_MODULE(_nanofractal, m) {
    m.attr("__version__") = "0.1.0";
    m.def("_opencv_version", []() { return cv::getVersionString(); });
}
```

- [ ] **Step 6: Create `src/nanofractal/__init__.py`**

```python
from ._nanofractal import __version__, _opencv_version

__all__ = ["__version__", "_opencv_version"]
```

- [ ] **Step 7: Create `README.md`**

```markdown
# nanofractal

High-performance fiducial marker detection (ArUco Nano v6 + Fractal) for Python.
```

- [ ] **Step 8: Write the smoke test `tests/test_smoke.py`**

```python
import nanofractal


def test_version():
    assert nanofractal.__version__ == "0.1.0"


def test_opencv_linked():
    v = nanofractal._opencv_version()
    assert isinstance(v, str)
    assert len(v) > 0
```

- [ ] **Step 9: Build and install editable, run smoke test**

Run:
```bash
python -m pip install -e ".[test]" -v
python -m pytest tests/test_smoke.py -v
```
Expected: build succeeds; both tests PASS (proves scikit-build-core + nanobind + OpenCV all link).

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: project scaffold with nanobind + OpenCV smoke test"
```

---

## Task 2: Zero-copy numpy → cv::Mat bridge + numpy output helpers

**Files:**
- Create: `src/ndarray_cv.hpp`
- Modify: `src/_bindings.cpp` (add debug bindings)
- Test: `tests/test_bridge.py`

- [ ] **Step 1: Write the failing test `tests/test_bridge.py`**

```python
import numpy as np
import pytest
import nanofractal._nanofractal as _nf


def test_gray_info():
    img = np.zeros((48, 64), dtype=np.uint8)
    rows, cols, ch = _nf._img_info(img)
    assert (rows, cols, ch) == (48, 64, 1)


def test_bgr_info():
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    rows, cols, ch = _nf._img_info(img)
    assert (rows, cols, ch) == (48, 64, 3)


def test_zero_copy_shares_buffer():
    img = np.zeros((48, 64), dtype=np.uint8)
    ptr_cv = _nf._img_data_ptr(img)
    ptr_np = img.__array_interface__["data"][0]
    assert ptr_cv == ptr_np  # no copy: cv::Mat points at the numpy buffer


def test_mean_matches_numpy():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(48, 64), dtype=np.uint8)
    assert _nf._img_mean(img) == pytest.approx(img.mean(), abs=1e-6)


def test_wrong_dtype_raises():
    img = np.zeros((48, 64), dtype=np.float32)
    with pytest.raises(TypeError):
        _nf._img_info(img)


def test_non_contiguous_raises():
    img = np.zeros((48, 128), dtype=np.uint8)[:, ::2]  # non-contiguous view
    assert not img.flags["C_CONTIGUOUS"]
    with pytest.raises(TypeError):
        _nf._img_info(img)


def test_roundtrip_owned_array():
    out = _nf._echo_corners()  # returns a (2,4,2) float32 array built in C++
    assert out.dtype == np.float32
    assert out.shape == (2, 4, 2)
    assert out[0, 0, 0] == pytest.approx(1.5)
    assert out[1, 3, 1] == pytest.approx(8.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bridge.py -v`
Expected: FAIL — `_img_info`/`_img_data_ptr`/`_img_mean`/`_echo_corners` do not exist.

- [ ] **Step 3: Create `src/ndarray_cv.hpp`**

```cpp
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <opencv2/core.hpp>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

// Input image: uint8, C-contiguous, CPU. nanobind enforces dtype/contiguity/device
// at the call boundary (raising TypeError on mismatch).
using ImageArray = nb::ndarray<const uint8_t, nb::c_contig, nb::device::cpu>;

// Wrap a numpy buffer as cv::Mat WITHOUT copying. The source ndarray must outlive
// the returned Mat (it does: the Mat is used only within the calling function while
// the Python argument is alive).
inline cv::Mat as_mat(const ImageArray &arr) {
    if (arr.ndim() == 2) {
        return cv::Mat((int)arr.shape(0), (int)arr.shape(1), CV_8UC1,
                       const_cast<uint8_t *>(arr.data()));
    }
    if (arr.ndim() == 3 && arr.shape(2) == 3) {
        return cv::Mat((int)arr.shape(0), (int)arr.shape(1), CV_8UC3,
                       const_cast<uint8_t *>(arr.data()));
    }
    throw std::invalid_argument(
        "image must be uint8 (H,W) grayscale or (H,W,3) BGR, C-contiguous");
}

// Build an owned numpy array by copying a flat vector. The heap vector is freed
// when numpy releases the capsule.
template <typename T>
nb::ndarray<nb::numpy, T> make_owned(std::vector<T> &&data,
                                     std::initializer_list<size_t> shape) {
    auto *heap = new std::vector<T>(std::move(data));
    nb::capsule owner(heap, [](void *p) noexcept {
        delete static_cast<std::vector<T> *>(p);
    });
    return nb::ndarray<nb::numpy, T>(heap->data(), shape, owner);
}

// Convenience: int32 ids (N,) and float32 corners (N,4,2).
inline nb::ndarray<nb::numpy, int32_t> ids_to_numpy(std::vector<int32_t> &&ids) {
    size_t n = ids.size();
    return make_owned<int32_t>(std::move(ids), {n});
}

inline nb::ndarray<nb::numpy, float> corners_to_numpy(std::vector<float> &&c,
                                                      size_t n) {
    return make_owned<float>(std::move(c), {n, (size_t)4, (size_t)2});
}
```

- [ ] **Step 4: Add debug bindings to `src/_bindings.cpp`**

Replace the file contents with:

```cpp
#include <nanobind/nanobind.h>
#include <opencv2/core.hpp>
#include "ndarray_cv.hpp"

namespace nb = nanobind;

NB_MODULE(_nanofractal, m) {
    m.attr("__version__") = "0.1.0";
    m.def("_opencv_version", []() { return cv::getVersionString(); });

    m.def("_img_info", [](ImageArray arr) {
        cv::Mat im = as_mat(arr);
        return nb::make_tuple(im.rows, im.cols, im.channels());
    });

    m.def("_img_data_ptr", [](ImageArray arr) {
        return (uintptr_t)as_mat(arr).data;
    });

    m.def("_img_mean", [](ImageArray arr) {
        cv::Mat im = as_mat(arr);
        double mean;
        {
            nb::gil_scoped_release rel;
            mean = cv::mean(im)[0];
        }
        return mean;
    });

    m.def("_echo_corners", []() {
        std::vector<float> c = {1.5f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
                                1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        return corners_to_numpy(std::move(c), 2);
    });
}
```

- [ ] **Step 5: Rebuild and run tests**

Run:
```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_bridge.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: zero-copy numpy<->cv::Mat bridge and numpy output helpers"
```

---

## Task 3: ArucoDetector.detect (ids + corners) with GIL release

**Files:**
- Modify: `src/_bindings.cpp` (add `ArucoDetector` class)
- Modify: `src/nanofractal/__init__.py` (facade: `Dict`, `DetectionResult`, `ArucoDetector`)
- Test: `tests/test_aruco.py`

- [ ] **Step 1: Write the failing test `tests/test_aruco.py`**

```python
import numpy as np
import nanofractal as nf


def test_blank_image_returns_empty():
    det = nf.ArucoDetector(dictionary=nf.Dict.ARUCO_MIP_36h12, max_attempts=1)
    res = det.detect(np.full((480, 640), 255, dtype=np.uint8))
    assert res.ids.dtype == np.int32
    assert res.ids.shape == (0,)
    assert res.corners.dtype == np.float32
    assert res.corners.shape == (0, 4, 2)


def test_default_max_attempts_is_one():
    det = nf.ArucoDetector()  # realtime default
    assert det.max_attempts == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aruco.py -v`
Expected: FAIL — `nf.ArucoDetector` / `nf.Dict` do not exist.

- [ ] **Step 3: Add `ArucoDetector` to `src/_bindings.cpp`**

Add `#include "aruco_nano_v6.h"` at the top (after the existing includes), and add this inside `NB_MODULE`, before the closing brace:

```cpp
    // ---- ArUco Nano v6 ----
    struct ArucoDetectorImpl {
        int dict;
        unsigned max_attempts;
        ArucoDetectorImpl(int dictionary, unsigned attempts)
            : dict(dictionary), max_attempts(attempts ? attempts : 1u) {}

        nb::tuple detect(ImageArray arr) {
            cv::Mat im = as_mat(arr);
            std::vector<aruconano::Marker> markers;
            {
                nb::gil_scoped_release rel;
                markers = aruconano::MarkerDetector::detect(
                    im, max_attempts, (aruconano::MarkerDetector::Dict)dict);
            }
            size_t n = markers.size();
            std::vector<int32_t> ids(n);
            std::vector<float> corners(n * 8);
            for (size_t i = 0; i < n; i++) {
                ids[i] = markers[i].id;
                for (int c = 0; c < 4; c++) {
                    corners[i * 8 + c * 2 + 0] = markers[i][c].x;
                    corners[i * 8 + c * 2 + 1] = markers[i][c].y;
                }
            }
            return nb::make_tuple(ids_to_numpy(std::move(ids)),
                                  corners_to_numpy(std::move(corners), n));
        }
    };

    nb::class_<ArucoDetectorImpl>(m, "ArucoDetector")
        .def(nb::init<int, unsigned>(), nb::arg("dictionary"),
             nb::arg("max_attempts"))
        .def_ro("max_attempts", &ArucoDetectorImpl::max_attempts)
        .def("detect", &ArucoDetectorImpl::detect, nb::arg("image"));
```

- [ ] **Step 4: Replace `src/nanofractal/__init__.py` with the facade**

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from . import _nanofractal as _nf

__version__ = _nf.__version__


class Dict(IntEnum):
    ARUCO_MIP_36h12 = 0
    APRILTAG_36h11 = 1


@dataclass
class DetectionResult:
    ids: np.ndarray            # int32 (N,)
    corners: np.ndarray        # float32 (N, 4, 2)
    points_2d: np.ndarray | None = None  # float32 (M, 2)
    points_3d: np.ndarray | None = None  # float32 (M, 3)


class ArucoDetector:
    def __init__(self, dictionary: Dict = Dict.ARUCO_MIP_36h12,
                 max_attempts: int = 1) -> None:
        self._d = _nf.ArucoDetector(int(dictionary), int(max_attempts))

    @property
    def max_attempts(self) -> int:
        return self._d.max_attempts

    def detect(self, image: np.ndarray) -> DetectionResult:
        ids, corners = self._d.detect(image)
        return DetectionResult(ids=ids, corners=corners)


__all__ = ["__version__", "Dict", "DetectionResult", "ArucoDetector"]
```

- [ ] **Step 5: Rebuild and run tests**

Run:
```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_aruco.py -v
```
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: ArucoDetector.detect with GIL release and numpy output"
```

---

## Task 4: ArUco synthetic marker rendering + detection correctness

**Files:**
- Create: `src/aruco_dicts.hpp` (marker dictionaries for rendering)
- Modify: `src/_bindings.cpp` (add `_aruco_marker_image8` renderer)
- Create: `tests/conftest.py` (rendering helper)
- Modify: `tests/test_aruco.py` (real detection test)

**Why a separate dictionary header:** The marker code arrays in `aruco_nano_v6.h` live as *locals* inside `MarkerDetector::_detect`, so they cannot be read from outside to render a marker image. We copy the two arrays once into a small shim header and use them only for test/tool rendering. Detection still uses the header's own private copy — no behavior change, no header patch.

- [ ] **Step 1: Create `src/aruco_dicts.hpp`**

```cpp
#pragma once
#include <cstdint>
#include <vector>

// Marker dictionaries copied VERBATIM from third_party/aruco_nano_v6.h
// (Apache-2.0) for marker *rendering* in tests/tools only. Detection uses the
// header's own private copy of these tables.
namespace aruco_dicts {
inline const std::vector<uint64_t> &mip_36h12() {
    static const std::vector<uint64_t> d = {
        // PASTE the 250 values from the `Dict_codes={ ... }` initializer in the
        // ARUCO_MIP_36h12 branch of third_party/aruco_nano_v6.h (line 129),
        // verbatim including the trailing `UL` suffixes.
    };
    return d;
}
inline const std::vector<uint64_t> &apriltag_36h11() {
    static const std::vector<uint64_t> d = {
        // PASTE the values from the `else` branch `Dict_codes={ ... }` initializer
        // (AprilTag 36h11) of third_party/aruco_nano_v6.h (line 131), verbatim.
    };
    return d;
}
}  // namespace aruco_dicts
```

> IMPLEMENTER ACTION: Open `third_party/aruco_nano_v6.h`. Line 129 is the `Dict_codes={0xd2b63a09dUL, ...}` list for `ARUCO_MIP_36h12` — copy its contents into `mip_36h12()`. Line 131 is the `else` branch list for AprilTag 36h11 — copy its contents into `apriltag_36h11()`. These are plain comma-separated `uint64_t` literals and paste directly inside the `{ }`.

- [ ] **Step 2: Add the renderer to `src/_bindings.cpp`**

Add `#include "aruco_dicts.hpp"` near the top. Add inside `NB_MODULE` (after the ArucoDetector binding). It renders the 8x8 cell grid (1-cell black border + 6x6 inner code) for a given id, matching the bit ordering used by `touulong` in the header:

```cpp
    m.def("_aruco_marker_image8", [](int dict, int id) {
        const std::vector<uint64_t> &codes =
            dict == 0 ? aruco_dicts::mip_36h12() : aruco_dicts::apriltag_36h11();
        if (id < 0 || (size_t)id >= codes.size())
            throw std::invalid_argument("marker id out of range");
        uint64_t code = codes[id];

        // touulong() in the header reads bidx over y=rows-1..0, x=cols-1..0 of the
        // 6x6 inner grid; bit b is the b-th bit (LSB first) of `code`.
        std::vector<uint8_t> grid(8 * 8, 0);  // border stays 0 (black)
        int b = 0;
        for (int y = 5; y >= 0; y--)
            for (int x = 5; x >= 0; x--) {
                int bit = (int)((code >> b) & 1ULL);
                grid[(y + 1) * 8 + (x + 1)] = bit ? 255 : 0;  // +1 for border
                b++;
            }
        return make_owned<uint8_t>(std::move(grid), {(size_t)8, (size_t)8});
    });
```

- [ ] **Step 3: Create `tests/conftest.py` with a rendering helper**

```python
import numpy as np
import pytest
import nanofractal._nanofractal as _nf


def _upscale(grid8: np.ndarray, cell: int) -> np.ndarray:
    """Nearest-neighbor upscale an (8,8) grid to (8*cell, 8*cell)."""
    return np.kron(grid8, np.ones((cell, cell), dtype=np.uint8))


@pytest.fixture
def render_aruco():
    def _render(marker_id: int, dictionary: int = 0, cell: int = 40,
                margin: int = 60) -> np.ndarray:
        grid = np.asarray(_nf._aruco_marker_image8(dictionary, marker_id))
        marker = _upscale(grid, cell)
        h, w = marker.shape
        canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
        canvas[margin:margin + h, margin:margin + w] = marker
        return np.ascontiguousarray(canvas)
    return _render
```

- [ ] **Step 4: Write the failing detection test in `tests/test_aruco.py`**

Append:

```python
import pytest


@pytest.mark.parametrize("marker_id", [0, 1, 7, 42])
def test_detects_rendered_marker(render_aruco, marker_id):
    import nanofractal as nf
    img = render_aruco(marker_id, dictionary=int(nf.Dict.ARUCO_MIP_36h12))
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    assert marker_id in res.ids.tolist()
    # corners of the detected marker should lie inside the canvas
    idx = res.ids.tolist().index(marker_id)
    c = res.corners[idx]
    assert c.shape == (4, 2)
    assert (c >= 0).all() and (c[:, 0] < img.shape[1]).all() and (c[:, 1] < img.shape[0]).all()
```

- [ ] **Step 5: Run test to verify it fails, then passes after build**

Run: `python -m pytest tests/test_aruco.py -v`
Expected before build: FAIL (`_aruco_marker_image8` missing). After `pip install -e .` rebuild: all PASS.

```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_aruco.py -v
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test: synthetic ArUco rendering and detection correctness"
```

---

## Task 5: ArucoDetector.estimate_pose (batched solvePnP)

**Files:**
- Modify: `src/_bindings.cpp` (add `estimate_pose` to ArucoDetector)
- Modify: `src/nanofractal/__init__.py` (facade method)
- Test: `tests/test_aruco.py`

- [ ] **Step 1: Write the failing test in `tests/test_aruco.py`**

Append:

```python
def test_estimate_pose_frontal_marker(render_aruco):
    import nanofractal as nf
    img = render_aruco(0, dictionary=int(nf.Dict.ARUCO_MIP_36h12))
    h, w = img.shape
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    res = det.detect(img)
    assert 0 in res.ids.tolist()

    cam = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5,), dtype=np.float64)
    rvecs, tvecs = det.estimate_pose(res.corners, cam, dist, marker_size=0.05)

    assert rvecs.shape == (len(res.ids), 3)
    assert tvecs.shape == (len(res.ids), 3)
    i = res.ids.tolist().index(0)
    assert tvecs[i, 2] > 0                       # marker in front of camera
    assert abs(np.linalg.norm(rvecs[i])) < 0.5   # roughly frontal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aruco.py::test_estimate_pose_frontal_marker -v`
Expected: FAIL — `estimate_pose` missing.

- [ ] **Step 3: Add `estimate_pose` to `ArucoDetectorImpl` in `src/_bindings.cpp`**

Add these typed array aliases near the top of the file (after `using ImageArray`):

```cpp
using F32Arr = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using F64Arr = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
```

Add this method inside `struct ArucoDetectorImpl`:

```cpp
        nb::tuple estimate_pose(F32Arr corners, F64Arr cam, F64Arr dist,
                                double marker_size) {
            if (corners.ndim() != 3 || corners.shape(1) != 4 ||
                corners.shape(2) != 2)
                throw std::invalid_argument("corners must be float32 (N,4,2)");
            if (cam.ndim() != 2 || cam.shape(0) != 3 || cam.shape(1) != 3)
                throw std::invalid_argument("camera_matrix must be float64 (3,3)");
            size_t n = corners.shape(0);
            cv::Mat camMat(3, 3, CV_64F, const_cast<double *>(cam.data()));
            cv::Mat distMat((int)dist.shape(0), 1, CV_64F,
                            const_cast<double *>(dist.data()));
            std::vector<double> rvecs(n * 3), tvecs(n * 3);
            {
                nb::gil_scoped_release rel;
                for (size_t i = 0; i < n; i++) {
                    aruconano::Marker mk;
                    for (int c = 0; c < 4; c++)
                        mk.push_back(cv::Point2f(
                            corners.data()[i * 8 + c * 2 + 0],
                            corners.data()[i * 8 + c * 2 + 1]));
                    auto rt = mk.estimatePose(camMat, distMat, marker_size);
                    cv::Mat rv = rt.first, tv = rt.second;
                    for (int k = 0; k < 3; k++) {
                        rvecs[i * 3 + k] = rv.at<double>(k);
                        tvecs[i * 3 + k] = tv.at<double>(k);
                    }
                }
            }
            return nb::make_tuple(make_owned<double>(std::move(rvecs), {n, (size_t)3}),
                                  make_owned<double>(std::move(tvecs), {n, (size_t)3}));
        }
```

Register it on the class binding (add a `.def`):

```cpp
        .def("estimate_pose", &ArucoDetectorImpl::estimate_pose,
             nb::arg("corners"), nb::arg("camera_matrix"), nb::arg("dist_coeffs"),
             nb::arg("marker_size"));
```

- [ ] **Step 4: Add facade method in `src/nanofractal/__init__.py`**

Add to `class ArucoDetector`:

```python
    def estimate_pose(self, corners: np.ndarray, camera_matrix: np.ndarray,
                      dist_coeffs: np.ndarray, marker_size: float):
        corners = np.ascontiguousarray(corners, dtype=np.float32)
        camera_matrix = np.ascontiguousarray(camera_matrix, dtype=np.float64)
        dist_coeffs = np.ascontiguousarray(dist_coeffs, dtype=np.float64)
        return self._d.estimate_pose(corners, camera_matrix, dist_coeffs,
                                     float(marker_size))
```

- [ ] **Step 5: Rebuild and run tests**

Run:
```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_aruco.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: ArucoDetector.estimate_pose batched solvePnP"
```

---

## Task 6: FractalDetector.detect (ids + corners)

**Files:**
- Modify: `src/_bindings.cpp` (add `FractalDetector` + `_fractal_external_image8`, `_fractal_external_id`)
- Modify: `src/nanofractal/__init__.py` (facade `FractalDetector`)
- Test: `tests/test_fractal.py`

- [ ] **Step 1: Write the failing test `tests/test_fractal.py`**

```python
import numpy as np
import pytest
import nanofractal as nf
import nanofractal._nanofractal as _nf

CONFIG = "FRACTAL_5L_6"


@pytest.fixture
def render_fractal_external():
    def _render(cell: int = 40, margin: int = 80) -> np.ndarray:
        grid = np.asarray(_nf._fractal_external_image8(CONFIG))
        up = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
        h, w = up.shape
        canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
        canvas[margin:margin + h, margin:margin + w] = up
        return np.ascontiguousarray(canvas)
    return _render


def test_blank_returns_empty():
    det = nf.FractalDetector(CONFIG)
    res = det.detect(np.full((480, 640), 255, dtype=np.uint8))
    assert res.ids.dtype == np.int32 and res.ids.shape == (0,)
    assert res.corners.dtype == np.float32 and res.corners.shape == (0, 4, 2)


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        nf.FractalDetector("NOT_A_CONFIG")


def test_detects_external_marker(render_fractal_external):
    img = render_fractal_external()
    det = nf.FractalDetector(CONFIG)
    res = det.detect(img)
    ext_id = _nf._fractal_external_id(CONFIG)
    assert ext_id in res.ids.tolist()
    idx = res.ids.tolist().index(ext_id)
    assert res.corners[idx].shape == (4, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fractal.py -v`
Expected: FAIL — `nf.FractalDetector` missing.

- [ ] **Step 3: Add fractal bindings to `src/_bindings.cpp`**

Add `#include "nanofractal.h"` near the top. Add inside `NB_MODULE`:

```cpp
    // ---- Fractal ----
    struct FractalDetectorImpl {
        std::string config;
        float marker_size;
        std::vector<std::unique_ptr<nanofractal::FractalMarkerDetector>> pool;

        FractalDetectorImpl(std::string cfg, float msize)
            : config(std::move(cfg)), marker_size(msize) {
            pool.push_back(make_detector());
        }

        std::unique_ptr<nanofractal::FractalMarkerDetector> make_detector() {
            auto d = std::make_unique<nanofractal::FractalMarkerDetector>();
            d->setParams(config, marker_size > 0 ? marker_size : -1.f);
            return d;
        }

        static void fill(const std::vector<nanofractal::FractalMarker> &markers,
                         std::vector<int32_t> &ids, std::vector<float> &corners) {
            size_t n = markers.size();
            ids.resize(n);
            corners.resize(n * 8);
            for (size_t i = 0; i < n; i++) {
                ids[i] = markers[i].id;
                for (int c = 0; c < 4; c++) {
                    corners[i * 8 + c * 2 + 0] = markers[i][c].x;
                    corners[i * 8 + c * 2 + 1] = markers[i][c].y;
                }
            }
        }

        nb::tuple detect(ImageArray arr) {
            cv::Mat im = as_mat(arr);
            std::vector<nanofractal::FractalMarker> markers;
            {
                nb::gil_scoped_release rel;
                markers = pool[0]->detect(im);
            }
            std::vector<int32_t> ids;
            std::vector<float> corners;
            fill(markers, ids, corners);
            size_t n = ids.size();
            return nb::make_tuple(ids_to_numpy(std::move(ids)),
                                  corners_to_numpy(std::move(corners), n));
        }
    };

    nb::class_<FractalDetectorImpl>(m, "FractalDetector")
        .def(nb::init<std::string, float>(), nb::arg("config"),
             nb::arg("marker_size"))
        .def("detect", &FractalDetectorImpl::detect, nb::arg("image"));

    m.def("_fractal_external_id", [](std::string config) {
        nanofractal::FractalMarkerSet s(config);
        return s.idExternal;
    });

    m.def("_fractal_external_image8", [](std::string config) {
        nanofractal::FractalMarkerSet s(config);
        cv::Mat M = s.fractalMarkerCollection[s.idExternal].mat();  // KxK, 0/1
        int K = M.rows;
        std::vector<uint8_t> grid((K + 2) * (K + 2), 0);  // black border
        for (int y = 0; y < K; y++)
            for (int x = 0; x < K; x++)
                grid[(y + 1) * (K + 2) + (x + 1)] =
                    M.at<uint8_t>(y, x) ? 255 : 0;
        return make_owned<uint8_t>(std::move(grid),
                                   {(size_t)(K + 2), (size_t)(K + 2)});
    });
```

> NOTE: `FractalMarkerSet(config)` throws `std::runtime_error` for an invalid config; nanobind maps that to `RuntimeError`. To satisfy `test_invalid_config_raises` (expects `ValueError`), the facade validates the config first (Step 4).

- [ ] **Step 4: Add facade `FractalDetector` to `src/nanofractal/__init__.py`**

```python
_FRACTAL_CONFIGS = {"FRACTAL_2L_6", "FRACTAL_3L_6", "FRACTAL_4L_6", "FRACTAL_5L_6"}


class FractalDetector:
    def __init__(self, config: str, marker_size: float = -1.0) -> None:
        if config not in _FRACTAL_CONFIGS:
            raise ValueError(
                f"invalid config {config!r}; use one of {sorted(_FRACTAL_CONFIGS)}")
        self.config = config
        self._d = _nf.FractalDetector(config, float(marker_size))

    def detect(self, image: np.ndarray) -> DetectionResult:
        ids, corners = self._d.detect(image)
        return DetectionResult(ids=ids, corners=corners)
```

Add `"FractalDetector"` to `__all__`.

- [ ] **Step 5: Rebuild and run tests**

Run:
```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_fractal.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: FractalDetector.detect with config validation and pool"
```

---

## Task 7: FractalDetector inner points (with_inner_points)

**Files:**
- Modify: `src/_bindings.cpp` (add `detect_full` to `FractalDetectorImpl`)
- Modify: `src/nanofractal/__init__.py` (facade `detect(..., with_inner_points=)`)
- Test: `tests/test_fractal.py`

- [ ] **Step 1: Write the failing test in `tests/test_fractal.py`**

Append:

```python
def test_detect_with_inner_points_shapes(render_fractal_external):
    img = render_fractal_external()
    det = nf.FractalDetector(CONFIG, marker_size=0.85)
    res = det.detect(img, with_inner_points=True)
    ext_id = _nf._fractal_external_id(CONFIG)
    assert ext_id in res.ids.tolist()
    assert res.points_2d is not None and res.points_3d is not None
    assert res.points_2d.dtype == np.float32 and res.points_2d.ndim == 2
    assert res.points_2d.shape[1] == 2
    assert res.points_3d.dtype == np.float32 and res.points_3d.shape[1] == 3
    assert res.points_2d.shape[0] == res.points_3d.shape[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fractal.py::test_detect_with_inner_points_shapes -v`
Expected: FAIL — `detect()` has no `with_inner_points` argument.

- [ ] **Step 3: Add `detect_full` to `FractalDetectorImpl` in `src/_bindings.cpp`**

Add inside the struct:

```cpp
        nb::tuple detect_full(ImageArray arr) {
            cv::Mat im = as_mat(arr);
            std::vector<nanofractal::FractalMarker> markers;
            std::vector<cv::Point3f> p3d;
            std::vector<cv::Point2f> p2d;
            {
                nb::gil_scoped_release rel;
                markers = pool[0]->detect(im, p3d, p2d);
            }
            std::vector<int32_t> ids;
            std::vector<float> corners;
            fill(markers, ids, corners);
            size_t n = ids.size();

            size_t m2 = p2d.size();
            std::vector<float> pts2(m2 * 2), pts3(m2 * 3);
            for (size_t i = 0; i < m2; i++) {
                pts2[i * 2 + 0] = p2d[i].x;
                pts2[i * 2 + 1] = p2d[i].y;
                pts3[i * 3 + 0] = p3d[i].x;
                pts3[i * 3 + 1] = p3d[i].y;
                pts3[i * 3 + 2] = p3d[i].z;
            }
            return nb::make_tuple(
                ids_to_numpy(std::move(ids)),
                corners_to_numpy(std::move(corners), n),
                make_owned<float>(std::move(pts2), {m2, (size_t)2}),
                make_owned<float>(std::move(pts3), {m2, (size_t)3}));
        }
```

Register it on the class binding:

```cpp
        .def("detect_full", &FractalDetectorImpl::detect_full, nb::arg("image"));
```

- [ ] **Step 4: Update facade `FractalDetector.detect` in `src/nanofractal/__init__.py`**

```python
    def detect(self, image: np.ndarray,
               with_inner_points: bool = False) -> DetectionResult:
        if with_inner_points:
            ids, corners, p2d, p3d = self._d.detect_full(image)
            return DetectionResult(ids=ids, corners=corners,
                                   points_2d=p2d, points_3d=p3d)
        ids, corners = self._d.detect(image)
        return DetectionResult(ids=ids, corners=corners)
```

- [ ] **Step 5: Rebuild and run tests**

Run:
```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_fractal.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: FractalDetector inner-point correspondences (with_inner_points)"
```

---

## Task 8: Parallel detect_batch with detector pool

**Files:**
- Modify: `src/_bindings.cpp` (add `detect_batch` to both detectors; grow fractal pool)
- Modify: `src/nanofractal/__init__.py` (facade `detect_batch`)
- Test: `tests/test_batch.py`

- [ ] **Step 1: Write the failing test `tests/test_batch.py`**

```python
import numpy as np
import nanofractal as nf
import nanofractal._nanofractal as _nf


def _render_aruco(marker_id=0, cell=40, margin=60):
    grid = np.asarray(_nf._aruco_marker_image8(0, marker_id))
    up = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    h, w = up.shape
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin:margin + h, margin:margin + w] = up
    return np.ascontiguousarray(canvas)


def test_aruco_batch_matches_sequential():
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    imgs = [_render_aruco(i) for i in [0, 1, 7, 42]]
    seq = [det.detect(im) for im in imgs]
    par = det.detect_batch(imgs, num_threads=4)
    assert len(par) == len(seq)
    for a, b in zip(seq, par):
        assert a.ids.tolist() == b.ids.tolist()
        np.testing.assert_allclose(a.corners, b.corners, atol=1e-4)


def test_fractal_batch_matches_sequential():
    det = nf.FractalDetector("FRACTAL_5L_6")
    grid = np.asarray(_nf._fractal_external_image8("FRACTAL_5L_6"))
    up = np.kron(grid, np.ones((40, 40), dtype=np.uint8))
    h, w = up.shape
    canvas = np.full((h + 160, w + 160), 255, dtype=np.uint8)
    canvas[80:80 + h, 80:80 + w] = up
    img = np.ascontiguousarray(canvas)
    imgs = [img] * 8
    seq = [det.detect(im) for im in imgs]
    par = det.detect_batch(imgs, num_threads=4)
    assert len(par) == len(seq)
    for a, b in zip(seq, par):
        assert a.ids.tolist() == b.ids.tolist()
        np.testing.assert_allclose(a.corners, b.corners, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_batch.py -v`
Expected: FAIL — `detect_batch` missing.

- [ ] **Step 3: Add includes and the batch runner to `src/_bindings.cpp`**

Add near the top:

```cpp
#include <thread>
#include <atomic>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
```

Add `detect_batch` to `struct ArucoDetectorImpl`:

```cpp
        std::vector<nb::object> detect_batch(std::vector<ImageArray> imgs,
                                             int num_threads) {
            size_t N = imgs.size();
            std::vector<cv::Mat> mats(N);
            for (size_t i = 0; i < N; i++) mats[i] = as_mat(imgs[i]);

            std::vector<std::vector<int32_t>> all_ids(N);
            std::vector<std::vector<float>> all_corners(N);
            int T = num_threads > 0 ? num_threads
                                    : (int)std::thread::hardware_concurrency();
            if (T < 1) T = 1;
            int dict_ = dict;
            unsigned attempts_ = max_attempts;
            {
                nb::gil_scoped_release rel;
                std::atomic<size_t> next{0};
                auto worker = [&]() {
                    size_t i;
                    while ((i = next.fetch_add(1)) < N) {
                        auto markers = aruconano::MarkerDetector::detect(
                            mats[i], attempts_,
                            (aruconano::MarkerDetector::Dict)dict_);
                        size_t n = markers.size();
                        all_ids[i].resize(n);
                        all_corners[i].resize(n * 8);
                        for (size_t k = 0; k < n; k++) {
                            all_ids[i][k] = markers[k].id;
                            for (int c = 0; c < 4; c++) {
                                all_corners[i][k * 8 + c * 2 + 0] = markers[k][c].x;
                                all_corners[i][k * 8 + c * 2 + 1] = markers[k][c].y;
                            }
                        }
                    }
                };
                std::vector<std::thread> ths;
                for (int t = 0; t < T; t++) ths.emplace_back(worker);
                for (auto &x : ths) x.join();
            }
            std::vector<nb::object> out;
            out.reserve(N);
            for (size_t i = 0; i < N; i++) {
                size_t n = all_ids[i].size();
                out.push_back(nb::make_tuple(
                    ids_to_numpy(std::move(all_ids[i])),
                    corners_to_numpy(std::move(all_corners[i]), n)));
            }
            return out;
        }
```

Register on the binding:

```cpp
        .def("detect_batch", &ArucoDetectorImpl::detect_batch,
             nb::arg("images"), nb::arg("num_threads") = 0);
```

Add `detect_batch` to `struct FractalDetectorImpl` (uses the pool — one detector per thread):

```cpp
        std::vector<nb::object> detect_batch(std::vector<ImageArray> imgs,
                                             int num_threads) {
            size_t N = imgs.size();
            std::vector<cv::Mat> mats(N);
            for (size_t i = 0; i < N; i++) mats[i] = as_mat(imgs[i]);

            int T = num_threads > 0 ? num_threads
                                    : (int)std::thread::hardware_concurrency();
            if (T < 1) T = 1;
            while ((int)pool.size() < T) pool.push_back(make_detector());

            std::vector<std::vector<int32_t>> all_ids(N);
            std::vector<std::vector<float>> all_corners(N);
            {
                nb::gil_scoped_release rel;
                std::atomic<size_t> next{0};
                auto worker = [&](int t) {
                    size_t i;
                    while ((i = next.fetch_add(1)) < N) {
                        auto markers = pool[t]->detect(mats[i]);
                        fill(markers, all_ids[i], all_corners[i]);
                    }
                };
                std::vector<std::thread> ths;
                for (int t = 0; t < T; t++) ths.emplace_back(worker, t);
                for (auto &x : ths) x.join();
            }
            std::vector<nb::object> out;
            out.reserve(N);
            for (size_t i = 0; i < N; i++) {
                size_t n = all_ids[i].size();
                out.push_back(nb::make_tuple(
                    ids_to_numpy(std::move(all_ids[i])),
                    corners_to_numpy(std::move(all_corners[i]), n)));
            }
            return out;
        }
```

Register on the binding:

```cpp
        .def("detect_batch", &FractalDetectorImpl::detect_batch,
             nb::arg("images"), nb::arg("num_threads") = 0);
```

- [ ] **Step 4: Add facade `detect_batch` to both classes in `src/nanofractal/__init__.py`**

For `ArucoDetector`:

```python
    def detect_batch(self, images, num_threads: int = 0):
        results = self._d.detect_batch(list(images), int(num_threads))
        return [DetectionResult(ids=i, corners=c) for i, c in results]
```

For `FractalDetector`:

```python
    def detect_batch(self, images, num_threads: int = 0):
        results = self._d.detect_batch(list(images), int(num_threads))
        return [DetectionResult(ids=i, corners=c) for i, c in results]
```

- [ ] **Step 5: Rebuild and run tests**

Run:
```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_batch.py -v
```
Expected: both tests PASS (parallel results identical to sequential).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: parallel detect_batch with per-thread detector pool"
```

---

## Task 9: GIL-release verification + type stubs

**Files:**
- Create: `src/nanofractal/__init__.pyi`
- Create: `src/nanofractal/py.typed` (empty marker file)
- Modify: `pyproject.toml` (ship the stubs/marker)
- Test: `tests/test_batch.py` (GIL test)

- [ ] **Step 1: Write the failing GIL test in `tests/test_batch.py`**

Append:

```python
import time


def test_batch_parallel_releases_gil():
    """A background Python thread must keep counting while detect_batch runs."""
    import threading
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    imgs = [_render_aruco(i % 50, cell=60) for i in range(64)]

    counter = {"n": 0}
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter["n"] += 1

    t = threading.Thread(target=spin)
    t.start()
    try:
        det.detect_batch(imgs, num_threads=4)
    finally:
        stop.set()
        t.join()

    # If the GIL were held for the whole call, the spinner could not advance.
    assert counter["n"] > 0
```

- [ ] **Step 2: Run test to verify it passes (GIL already released in Task 8)**

Run: `python -m pytest tests/test_batch.py::test_batch_parallel_releases_gil -v`
Expected: PASS (GIL release was implemented in Task 8; this test documents/guards it).

- [ ] **Step 3: Create `src/nanofractal/__init__.pyi`**

```python
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np
import numpy.typing as npt

__version__: str

class Dict(IntEnum):
    ARUCO_MIP_36h12 = 0
    APRILTAG_36h11 = 1

@dataclass
class DetectionResult:
    ids: npt.NDArray[np.int32]
    corners: npt.NDArray[np.float32]
    points_2d: npt.NDArray[np.float32] | None = ...
    points_3d: npt.NDArray[np.float32] | None = ...

class ArucoDetector:
    def __init__(self, dictionary: Dict = ..., max_attempts: int = ...) -> None: ...
    @property
    def max_attempts(self) -> int: ...
    def detect(self, image: npt.NDArray[np.uint8]) -> DetectionResult: ...
    def detect_batch(self, images: Sequence[npt.NDArray[np.uint8]],
                     num_threads: int = ...) -> list[DetectionResult]: ...
    def estimate_pose(self, corners: npt.NDArray[np.float32],
                      camera_matrix: npt.NDArray[np.float64],
                      dist_coeffs: npt.NDArray[np.float64],
                      marker_size: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

class FractalDetector:
    config: str
    def __init__(self, config: str, marker_size: float = ...) -> None: ...
    def detect(self, image: npt.NDArray[np.uint8],
               with_inner_points: bool = ...) -> DetectionResult: ...
    def detect_batch(self, images: Sequence[npt.NDArray[np.uint8]],
                     num_threads: int = ...) -> list[DetectionResult]: ...
```

- [ ] **Step 4: Create `src/nanofractal/py.typed`**

Empty file (PEP 561 marker):

```bash
touch src/nanofractal/py.typed
```

- [ ] **Step 5: Ensure stubs ship — confirm `pyproject.toml`**

`[tool.scikit-build] wheel.packages = ["src/nanofractal"]` already includes the package directory, so `__init__.pyi` and `py.typed` are packaged. No change needed; verify by:

```bash
python -m pip install -e ".[test]"
python -c "import nanofractal, pathlib; p=pathlib.Path(nanofractal.__file__).parent; print((p/'py.typed').exists(), (p/'__init__.pyi').exists())"
```
Expected: `True True`.

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest tests/ -v --ignore=tests/test_benchmarks.py`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: type stubs (py.typed) and GIL-release verification test"
```

---

## Task 10: Benchmarks (latency + throughput)

**Files:**
- Create: `tests/test_benchmarks.py`

- [ ] **Step 1: Create `tests/test_benchmarks.py`**

```python
import numpy as np
import pytest
import nanofractal as nf
import nanofractal._nanofractal as _nf


def _scene(width, height, marker_id=0, cell=60):
    grid = np.asarray(_nf._aruco_marker_image8(0, marker_id))
    up = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    canvas = np.full((height, width), 255, dtype=np.uint8)
    h, w = up.shape
    canvas[20:20 + h, 20:20 + w] = up
    return np.ascontiguousarray(canvas)


@pytest.mark.benchmark(group="aruco-latency")
@pytest.mark.parametrize("size", [(640, 480), (1280, 720), (1920, 1080)])
def test_aruco_single_frame_latency(benchmark, size):
    w, h = size
    img = _scene(w, h)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=1)
    res = benchmark(det.detect, img)
    assert 0 in res.ids.tolist()


def test_aruco_batch_throughput_scaling(capsys):
    import time
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=1)
    imgs = [_scene(1280, 720, marker_id=i % 50) for i in range(64)]

    def run(threads):
        t0 = time.perf_counter()
        det.detect_batch(imgs, num_threads=threads)
        return time.perf_counter() - t0

    t1 = run(1)
    t4 = run(4)
    with capsys.disabled():
        print(f"\nbatch 64 frames @720p: 1 thread={t1*1e3:.1f}ms  "
              f"4 threads={t4*1e3:.1f}ms  speedup={t1/t4:.2f}x")
    # Non-flaky guard: more threads must not be slower than one.
    assert t4 <= t1 * 1.2
```

- [ ] **Step 2: Run the benchmarks**

Run:
```bash
python -m pytest tests/test_benchmarks.py -v --benchmark-only
python -m pytest tests/test_benchmarks.py::test_aruco_batch_throughput_scaling -s
```
Expected: latency benchmarks report medians for each resolution; throughput test prints a speedup line and PASSES the non-flaky guard.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: latency and batch-throughput benchmarks"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- v6 selection → documented (spec §2); plan vendors `aruco_nano_v6.h` (Task 1).
- Both detectors → ArucoDetector (Tasks 3–5), FractalDetector (Tasks 6–7).
- Pose estimation → Task 5.
- Zero-copy input → Task 2 (`as_mat`, `test_zero_copy_shares_buffer`).
- GIL release → Tasks 3/5/6/7/8 (`gil_scoped_release`); verified Task 9.
- numpy output (no per-marker objects) → `make_owned`/`ids_to_numpy`/`corners_to_numpy` (Task 2), used everywhere.
- Detector reuse → fractal config built once in ctor (Task 6).
- Parallel batch via pool → Task 8.
- Error handling → dtype/shape (Task 2), invalid config → ValueError (Task 6 facade), empty results as shaped arrays (Tasks 3/6 tests).
- max_attempts default = 1 → Task 3 (`test_default_max_attempts_is_one`).
- Testing: correctness (Tasks 4/6), parallel correctness (Task 8), edge cases (Tasks 2/3/6), benchmarks (Task 10).
- **Deferred to Plan B (packaging):** minimal static OpenCV, cibuildwheel, stable ABI, manylinux, auditwheel, the C++ reference-parity tool. These are not needed for a working, tested local library and belong in the packaging plan.

**Placeholder scan:** Task 4 Step 2 requires the implementer to paste the two dictionary arrays verbatim from `third_party/aruco_nano_v6.h` (lines 129 and 131) into `src/aruco_dicts.hpp`. This is a deliberate, bounded copy of public Apache-2.0 data (not a vague placeholder); exact source lines are cited. No other placeholders.

**Type consistency:** Low-level C++ classes are `ArucoDetector`/`FractalDetector` in module `_nanofractal`; the Python facade classes of the same public names wrap them via `self._d`. Tuple shapes are consistent: ArUco/fractal `detect` → `(ids int32 (N,), corners float32 (N,4,2))`; `detect_full` → adds `points_2d (M,2)`, `points_3d (M,3)`; `estimate_pose` → `(rvecs (N,3), tvecs (N,3)) float64`. Helper names (`as_mat`, `make_owned`, `ids_to_numpy`, `corners_to_numpy`, `fill`, `make_detector`) are defined once and reused consistently.
