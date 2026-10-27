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
| `detect_batch(images, num_threads=0, roi=None)` | `list[DetectionResult]` (no inner points) |
| `estimate_pose(result, camera_matrix, dist_coeffs, fisheye=False, return_method=False)` | `(rvec, tvec, reproj_err)` or `None` |
| `draw(image, result, camera_matrix=None, dist_coeffs=None, rvec=None, tvec=None, axis_length=None, inplace=True)` | the image drawn on |

- `image`: `uint8` C-contiguous `(H,W)`, `(H,W,1)` or `(H,W,3)` BGR.
- `roi`: `(x, y, width, height)` inside the image; results are in full-image
  coordinates.
- `num_threads=0`: all cores.
- `estimate_pose`: `camera_matrix` 3×3, `dist_coeffs` of length 4/5/8/12/14
  (exactly 4 with `fisheye=True`); uses inner points when the result has ≥ 4,
  else the outer corners of every detected level; `return_method=True` adds
  `"inner"` or `"corners"` as a 4th element.
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

`len(r)`, `bool(r)`, `for level_id, corners in r`, readable `repr`.

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
| `PoseSmoother(process_noise=1e-4, measurement_noise=1e-2, mode="kalman")` | `.update(rvec, tvec) -> (rvec, tvec)`, `.reset()`; `mode="ema"` for a fixed gain |

`nanofractal.__version__` is the package version.

## Command line

```text
python -m nanofractal.bench [--resolution WxH] [--scene clean|perspective|clutter]
                            [--mode detect|inner|scale05] [--config FRACTAL_5L_6] [--frames N]
```
