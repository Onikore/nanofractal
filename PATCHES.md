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

## third_party/nanofractal.h — round the submarker mask extent

**Symptom:** with `-ffast-math` (which the CI wheels enable via `NF_MARCH`),
level 3 of `FRACTAL_5L_6` was never detected, even on a perfect render.

**Root cause:** `FractalMarker::addSubFractalMarker` computes the submarker's
extent in parent bits as a float (`nsubBits = subSize / bitSize`) and then
truncates it: `int x_max = x_min + nsubBits`. The value is an exact integer in
theory, but float division can land just below it (fast-math turns the
division into a reciprocal multiply: 6.0 → 5.9999994). The mask then covered
one cell too few, so the decoder compared the parent's bits under the child
marker and rejected the level.

**Patch:** `std::lround(nsubBits)` instead of truncation, marked with a
`// nanofractal patch:` comment.

Regression test: `tests/test_generate.py::test_nested_round_trip_detects_every_level`
(every level of every config is detected on a nested render).

## third_party/nanofractal.h — bounded reads in getSubpixelValue

**Symptom:** AddressSanitizer: heap-buffer-overflow reading the input image,
e.g. on a 333×1001 (taller than wide) grayscale frame.

**Root cause:** the bilinear sample reads `(x+1, y)`, `(x, y+1)` and
`(x+1, y+1)` with no upper-bound check, and the row clamp compares with the
image *width* (`if(tl.y>=im_grey.cols) tl.y=im_grey.rows-1;`), so a sample
near the bottom of a tall image can read whole rows past the buffer.

**Patch:** the four reads go through a helper that clamps the linear index to
the last pixel, marked with a `// nanofractal patch:` comment. In-bounds
results are unchanged; the optimised pipeline (`src/fractal_fast.hpp`) uses
the identical clamp, so the two stay bit-identical. Found by the CI ASan job
running `tests/test_reference_equivalence.py::test_bgr_and_odd_sizes_identical`.
