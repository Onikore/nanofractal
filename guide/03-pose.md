# 3. Pose

## What you need

- **Camera intrinsics** `K` (3×3, float64) and **distortion coefficients**
  `D` (4, 5, 8, 12 or 14 values) — from a calibration, e.g. OpenCV's
  `cv2.calibrateCamera` with a chessboard. Guessed intrinsics give a pose with
  the right shape but wrong scale/angles.
- **`marker_size`** — the side of the external marker's black square, in the
  unit you want the translation in (metres below). Set it when creating the
  detector; without it the pose has no metric scale.

## Pose of the marker

```python
import numpy as np
import nanofractal as nf

det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.20)
page = np.full((900, 900), 255, np.uint8)
page[100:800, 100:800] = nf.generate_fractal("FRACTAL_5L_6", 700)
K = np.array([[900.0, 0, 450], [0, 900.0, 450], [0, 0, 1]])
D = np.zeros(5)

res = det.detect(page, with_inner_points=True)
pose = det.estimate_pose(res, K, D)
if pose is not None:
    rvec, tvec, reproj_err = pose
    print("rvec", rvec, "tvec", tvec, "rms px", reproj_err)
```

`estimate_pose` returns `None` when no marker is visible, otherwise
`(rvec, tvec, reproj_err)`:

- `rvec`, `tvec` — float64 `(3,)`, the marker frame in camera coordinates
  (Rodrigues rotation; `cv2.Rodrigues(rvec)` gives the 3×3 matrix);
- `reproj_err` — RMS reprojection error in pixels. Use it to **reject bad
  poses**: values well above ~2 px mean a wrong match, motion blur or bad
  intrinsics.

The marker frame has its origin at the **centre of the marker**, X to the
right and Y up as printed, Z pointing out of the marker towards the camera.

### Inner corners vs outer corners

| | `detect(image)` | `detect(image, with_inner_points=True)` |
|---|---|---|
| correspondences | 4 outer corners of every detected level | every matched inner and outer corner of all visible levels (often 50–200) |
| cost (1080p) | ~1 ms | ~2–2.5 ms |
| pose | fine when the marker is large and fully visible | much more stable at distance, at steep angles and under occlusion |

Without inner points `estimate_pose` uses the four outer corners of **every
detected level**, each at its real position and size within the marker, so
the distance is right even when only an inner level is visible (the camera is
close and the external border is out of frame). You never need to handle the
"no inner points" case yourself:

```python
close = np.ascontiguousarray(page[350:550, 350:550])     # only the inner levels
Kc = np.array([[900.0, 0, 100], [0, 900.0, 100], [0, 0, 1]])
p = det.estimate_pose(det.detect(close), Kc, D, full=True)
print(p.method, p.tvec[2])   # "corners", the same distance as for the full page
```

### Robustness built in

- **Outliers.** With 8 or more correspondences, points whose reprojection
  error exceeds `max(1.5 px, 3 × median)` are dropped once and the pose is
  solved again, if at least 8 points and 60 % remain. `reproj_err` is then
  the RMS over the kept points. With half or more of the points wrong the
  pass cannot tell good from bad — the large `reproj_err` shows it.
- **A marker facing the camera exactly** (a camera looking straight down):
  OpenCV's IPPE solver degenerates there. A first solution worse than 1 px
  is solved again with `SOLVEPNP_SQPNP` and the better pose is kept.

### `full=True`

`estimate_pose(..., full=True)` returns a `Pose` named tuple:

| field | meaning |
|---|---|
| `rvec`, `tvec`, `reproj_err` | as in the default 3-tuple |
| `method` | `"inner"` (inner + outer points) or `"corners"` (level corners only) |
| `n_points`, `n_inliers` | correspondences before / after the outlier pass |
| `reproj_err_rel` | `reproj_err` divided by the side in pixels of the largest detected level |

`reproj_err_rel` is the better quality gate when the marker's size in the
frame varies a lot (landing, approach): 1 px is a bad pose for a marker 40 px
wide but excellent for one 800 px wide.

```python
p = det.estimate_pose(res, K, D, full=True)
if p is not None and p.reproj_err_rel < 0.005 and p.n_inliers >= 8:
    rvec, tvec = p.rvec, p.tvec
```

The correspondences are available if you prefer your own solver:

```python
print(res.points_2d.shape, res.points_3d.shape)   # (M,2) pixels, (M,3) metres, z = 0
print(res.points_level[:8])                       # (M,) level id of each point
```

`points_level` says which level each point belongs to — e.g. to weight the
small inner levels differently, or to solve from one level only:

```python
inner = res.points_level == res.ids.max()          # the innermost level found
print(inner.sum(), "points on level", res.ids.max())
```

## Fisheye lenses

For OpenCV's fisheye model pass its 4 coefficients `(k1, k2, k3, k4)` and
`fisheye=True`; points are undistorted with the fisheye model before solving
and the reprojection error is measured against the original pixels.

```python
D_fisheye = np.array([-0.02, 0.003, 0.0, 0.0])
pose = det.estimate_pose(res, K, D_fisheye, fisheye=True)
```

## Drawing

```python
canvas = np.ascontiguousarray(np.dstack([page] * 3))   # BGR, writable
det.draw(canvas, res)                                   # outlines + ids, in place
rvec, tvec, err = det.estimate_pose(res, K, D)
det.draw(canvas, res, K, D, rvec, tvec)                 # + pose axes
copy = det.draw(page, res, inplace=False)               # draw on a copy
```

## Smoothing a pose stream

Per-frame poses jitter by a fraction of a pixel's worth of angle.
`PoseSmoother` filters the pose with a Kalman filter (or an equivalent
fixed-gain EMA, `mode="ema"`). The orientation is kept as a quaternion, so a
camera looking straight down at the marker (rotation ≈ π, where OpenCV's
`rvec` flips sign from frame to frame) gives a smooth, continuous `rvec`.

```python
smoother = nf.PoseSmoother(process_noise=1e-4, measurement_noise=1e-2)  # mode="kalman"
for _ in range(3):
    rvec_s, tvec_s = smoother.update(rvec, tvec)
smoother.reset()   # call when the marker is lost for a while
```

Raise `process_noise` (or lower `measurement_noise`) to follow fast motion
with less lag; lower it for steadier output.

**For control loops pass the frame time.** With `t` (seconds) the filter uses
the real interval between frames (dropped frames are handled) and a
constant-velocity model, so a moving marker is followed without lag; here
`process_noise` is per second. Pass `reproj_err` and the number of points to
trust good poses more, and `gate` to drop outliers:

```python
smoother = nf.PoseSmoother(process_noise=1e-3, measurement_noise=1e-4, gate=4.7)
for i in range(5):
    t = i / 30                                         # e.g. the capture timestamp
    rvec_s, tvec_s = smoother.update(rvec, tvec, t=t, reproj_err=err, n_points=len(res.points_2d))
    if smoother.last_rejected:
        pass                                           # outlier: the prediction was returned
```

With `reproj_err` / `n_points`, `measurement_noise` is the variance of a
4-corner pose at 1 px RMS; it is scaled by `reproj_err**2 * 4 / n_points`.
A measurement farther than `gate` (Mahalanobis distance; 4.7 ≈ 99.9 % for 6
components) from the prediction is rejected; after `max_rejects=3` rejections
in a row the filter restarts from the measurement.
