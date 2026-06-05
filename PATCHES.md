# Patches to vendored headers

We vendor `third_party/aruco_nano_v6.h` and `third_party/nanofractal.h` from
upstream and keep them as close to upstream as possible. The unmodified upstream
of `nanofractal.h` is recoverable from git history (the commit immediately before
the patch described below). The only intentional divergences are listed here.

## third_party/nanofractal.h — guard the inner-point path against empty FAST keypoints

**Symptom:** `FractalMarkerDetector::detect(img, p3d, p2d)` (the `with_inner_points`
path) segfaulted on valid `uint8` images when `cv::FastFeatureDetector` returned
zero keypoints (e.g. clean/synthetic markers with no sub-cell texture).

**Root cause:** `_private::kfilter()` did `kpoints[0].response` without checking
for an empty vector; downstream, the picoflann kdtree `radiusSearch` would also
index an empty index.

**Patch (2 guards):**
1. In `kfilter()`: `if (kpoints.empty()) return;` before touching `kpoints[0]`.
2. In `detect(img, p3d, p2d)`, after `assignClass`: `if (kpoints.empty()) return detected;`
   so the kdtree matching is skipped and the markers are returned with empty
   `p2d`/`p3d`.

Both are marked inline with `// nanofractal patch:` comments. With no keypoints
there are simply no inner-corner correspondences, which is the correct result.

Regression tests: `tests/test_fractal.py::test_detect_with_inner_points_empty_is_safe`
(clean image → empty, no crash) and `::test_detect_with_inner_points_nonempty`
(noisy image → populated correspondences).

## Both headers — remove unused `#include <opencv2/highgui.hpp>`

`aruco_nano_v6.h` and `nanofractal.h` each `#include <opencv2/highgui.hpp>`, but
their detection/pose/draw code paths call **no** highgui functions (highgui only
appears in the example snippets in the file comments, e.g. `imread`/`imwrite`).
The include is removed (replaced by a `// nanofractal patch:` comment) so the
library builds against a **minimal OpenCV** without the `highgui` module — which
is how the portable wheels are built in CI (`ci/build-opencv.sh`). Builds against
a full system OpenCV are unaffected.
