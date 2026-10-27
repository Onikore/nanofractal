# 7. API reference

Everything below is importable from `nanofractal` (`import nanofractal as nf`).

## FractalDetector

```text
FractalDetector(config, marker_size=-1.0, params=None)
```

- `config` — `"FRACTAL_2L_6"`, `"FRACTAL_3L_6"`, `"FRACTAL_4L_6"` or
  `"FRACTAL_5L_6"`; anything else raises `ValueError`.
- `marker_size` — side of the external marker's black square in your length
  unit; needed for a metric pose (`-1`: no scale).
- `params` — a `DetectorParams` (copied; read and change it via `.params`).

| method | returns |
|---|---|
| `detect(image, with_inner_points=False, roi=None)` | `DetectionResult` |
| `diagnose(image, roi=None)` | `Diagnosis(result, n_contours, n_not_quad, quads)` — see [Troubleshooting](08-troubleshooting.md#why-a-candidate-was-dropped) |
| `detect_batch(images, num_threads=0, roi=None)` | `list[DetectionResult]` (no inner points) |
| `estimate_pose(result, camera_matrix, dist_coeffs, fisheye=False, full=False)` | `(rvec, tvec, reproj_err)`, a `Pose` with `full=True`, or `None` |
| `draw(image, result, camera_matrix=None, dist_coeffs=None, rvec=None, tvec=None, axis_length=None, inplace=True)` | the image drawn on |

- `image`: `uint8` C-contiguous `(H,W)`, `(H,W,1)` or `(H,W,3)` BGR.
- `roi`: `(x, y, width, height)` inside the image; results are in full-image
  coordinates.
- `num_threads=0`: all cores.
- `estimate_pose`: `camera_matrix` 3×3, `dist_coeffs` of length 4/5/8/12/14
  (exactly 4 with `fisheye=True`); uses inner points when the result has ≥ 4,
  else the outer corners of every detected level; one outlier pass with ≥ 8
  points; `full=True` returns `Pose(rvec, tvec, reproj_err, method, n_points,
  n_inliers, reproj_err_rel)`.
- `draw` needs a writable contiguous image when `inplace=True`; `axis_length`
  defaults to half `marker_size`.

Attributes: `config`, `marker_size`, `params`.

## FractalTracker

```text
FractalTracker(detector, margin=0.25, pad=16)
```

| member | |
|---|---|
| `detect(image, with_inner_points=False)` | like `detector.detect`, searching near the previous result |
| `reset()` | forget the previous result |
| `full_frame_detections` | how many frames needed a whole-frame search |

## DetectionResult

| attribute | type |
|---|---|
| `ids` | int32 `(N,)` — level ids, 0 = external |
| `corners` | float32 `(N, 4, 2)` — marker TL, TR, BR, BL in pixels |
| `points_2d` | float32 `(M, 2)` or `None` — inner+outer image points |
| `points_3d` | float32 `(M, 3)` or `None` — matching marker points, z = 0 |
| `points_level` | int32 `(M,)` or `None` — level id of each point |
| `multiple_instances` | `bool` — a level was seen in two separate places (two copies of the marker); only the largest of each level is returned |
| `inconsistent_ids` | int32 `(K,)` — levels dropped because they are not where the other levels put them |

`len(r)`, `bool(r)`, `for level_id, corners in r`, readable `repr`.

## Diagnosis

| attribute | type |
|---|---|
| `result` | the `DetectionResult` (same as `detect`) |
| `n_contours` | contours traced (at least `min_contour_size` long) |
| `n_not_quad` | of those, not a convex quadrilateral |
| `quads` | `list[QuadInfo]` |

`QuadInfo(corners, reason, cells, nearest_id, border_errors, bit_errors,
contrast)`: corners float32 `(4, 2)` before sub-pixel refinement; `reason`
one of `"accepted"`, `"duplicate"`, `"border"`, `"bits"`, `"inconsistent"`; `cells` the grid
size (border included) and `nearest_id` the code it came closest to;
`border_errors` frame cells read white; `bit_errors` inner bits that differ
from `nearest_id`; `contrast` mean bright minus mean dark cell (grey levels).

## DetectorParams

Fields `min_contour_size`, `adaptive_block_size`, `adaptive_c`,
`approx_poly_rate`, `subpix_win_size`, `kfilter_min_dist`,
`detection_scale` — see [Tuning](06-tuning.md).

## Functions

| function | |
|---|---|
| `generate_fractal(config, size_px=400, nested=True)` | uint8 `(size_px, size_px)` marker image |
| `refine_corners(image, corners, win_size=5)` | float32 `(N, 4, 2)` refined corners |
| `set_num_threads(n)` / `get_num_threads()` | threads per `detect()` call (`n <= 0`: all cores) |
| `PoseSmoother(process_noise=1e-4, measurement_noise=1e-2, mode="kalman", gate=None, max_rejects=3)` | `.update(rvec, tvec, t=None, reproj_err=None, n_points=None) -> (rvec, tvec)`, `.reset()`, `.last_rejected`; `mode="ema"` for a fixed gain |

`nanofractal.__version__` is the package version.

## Command line

```text
python -m nanofractal.bench [--resolution WxH] [--scene clean|perspective|clutter]
                            [--mode detect|inner|scale05] [--config FRACTAL_5L_6] [--frames N]
```
