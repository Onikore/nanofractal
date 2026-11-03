# 4. Video and cameras

## FractalTracker

In video the marker moves little between frames. `FractalTracker` wraps a
detector and searches only the region around the previous frame's markers
(grown by `margin` × their size + `pad` pixels). It falls back to the whole
frame whenever nothing was found before, the region finds nothing, or it finds
fewer levels than the previous frame — so a marker is never lost just because
the box was tight. Results are the same as `detect()`; measured on a moving
marker: **~2× faster at 4K, 1.4–1.8× at 1080p**, with identical corner
accuracy.

```python
import numpy as np
import nanofractal as nf

det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.20)
tracker = nf.FractalTracker(det, margin=0.25, pad=16)

# simulated video: the marker drifts to the right
marker = nf.generate_fractal("FRACTAL_5L_6", 300)
for i in range(5):
    frame = np.full((720, 1280), 255, np.uint8)
    frame[200:500, 100 + 60 * i:400 + 60 * i] = marker
    res = tracker.detect(frame)            # same arguments as detector.detect
    print(i, res.ids.tolist())

print("whole-frame searches:", tracker.full_frame_detections)
tracker.reset()                            # e.g. after a scene cut
```

`tracker.detect(frame, with_inner_points=True)` works the same way for pose.

### Shrinking the region (`auto_scale`)

With `auto_scale=True` the tracker also sets `detection_scale` for the region
from the previous frame: the largest level is shrunk to about
`scale_target_px` (400 by default) pixels, in a few fixed steps down to 0.35.
Corners are still refined on the full-resolution frame, so they stay as
accurate as without scaling; the smallest inner levels may drop out of
`ids` (inner points of every level are still found). On a 4K video of a
moving 500–1200 px marker: **3.7 ms per frame instead of 21 ms on one core
(×5.7; the plain tracker ×2.5), ×3.9 on all cores**, outer corners as
accurate as full detection (largest error against the true corners 0.517 px
vs 0.518 px).

```python
tracker = nf.FractalTracker(det, auto_scale=True)
for i in range(3):
    frame = np.full((720, 1280), 255, np.uint8)
    frame[200:500, 100 + 60 * i:400 + 60 * i] = marker
    print(i, tracker.detect(frame).ids.tolist())
```

### Keeping a marker at the edge of range (`use_prior`)

Far away (a few pixels per cell), with blur or a shadow across a cell, a
marker can flicker: read in one frame, one or two cells off in the next.
`use_prior=True` hands the last result to `detect(..., prior=...)`: a quad
where a level was a moment ago is compared with that level's code only and
accepted with a few wrong cells. Such levels report `bit_errors > 0`; every
level also reports `cell_px` and `contrast` — use them to weigh a detection
instead of a yes/no. Without a prior nothing changes.

```python
det3 = nf.FractalDetector("FRACTAL_3L_6", marker_size=0.18)
tracker = nf.FractalTracker(det3, use_prior=True)
frame = np.full((720, 1280), 255, np.uint8)
frame[200:480, 100:380] = nf.generate_fractal("FRACTAL_3L_6", 280)  # 20 px/cell
res = tracker.detect(frame)
print(res.ids.tolist(), res.cell_px.round(1).tolist(), res.bit_errors.tolist())

# two cells of the outer level now read wrong (e.g. a shadow edge)
bad = frame.copy()
bad[240:260, 160:180] ^= 255               # outer grid cell (2, 3)
bad[240:260, 200:220] ^= 255               # outer grid cell (2, 5)
print(det3.detect(bad).ids.tolist())       # plain: the outer level is gone
res = tracker.detect(bad)                  # kept, marked by bit_errors
print(res.ids.tolist(), res.bit_errors.tolist())
assert 0 in res.ids and 0 not in det3.detect(bad).ids
```

With your own loop, pass the last non-empty result:
`det.detect(frame, prior=last)`. Only grids of 10+ cells are read this way
(the outer levels); small inner codes are too close to random patterns.

### Picking up a marker before it is read (`capture_frames`)

On the way down, the marker is often in view long before it is first read:
2.5–3 px per cell, overexposed, a few cells off. There is no prior yet, so
`use_prior` cannot help. `capture_frames=N` lists, on frames where nothing
is detected, the quads that come as close to a level's code as a prior
allows; when such a candidate of the same level stays in place for `N - 1`
frames in a row, the next frame reads it with the candidate as prior. The
first detection is then the `N`-th frame. It implies `use_prior`.

```python
small = np.full((240, 320), 255, np.uint8)
small[60:130, 100:170] = nf.generate_fractal("FRACTAL_3L_6", 70)  # 5 px/cell
small[70:75, 115:120] ^= 255               # outer grid cell (2, 3)
small[70:75, 125:130] ^= 255               # outer grid cell (2, 5)
print(det3.detect(small).ids.tolist())     # plain: nothing

tracker = nf.FractalTracker(det3, capture_frames=3)
got = [tracker.detect(small).ids.tolist() for _ in range(4)]
print(got)                                 # [[], [], [0], [0]]
assert got == [[], [], [0], [0]]
```

Flight videos (3–4 m, an overexposed print): the marker was read 8–39 s
earlier than with `use_prior` alone; in 1242 frames of a flight with no
readable marker, and 1200 synthetic frames of marker-like objects, not one
candidate. While there is no track it costs 4–22 % of a `detect` on a
cluttered frame (+0.06 ms at 640×480, +0.77 ms at 1920×1080, one core).

### When the size is known (`expected_side_px`)

A rangefinder or a barometer gives the height, and so the marker's side in
the image: `side ≈ size · f / height` (metres, focal length in pixels).
`detect(..., expected_side_px=(lo, hi))` (also `tracker.detect`) uses it:

- quads of a level's expected size are read with a looser tolerance — up to
  3 border cells and `sz² / 10` data cells off (14 of 144 for the outer level
  of `FRACTAL_3L_6`, against 8 with `prior`), and a lower cell contrast (30
  grey levels instead of 40: an overexposed print);
- the minimum contour drops to fit the expected size (a 28 px marker has a
  perimeter under the default 120);
- quads of that size that still do not read, but come within `sz² / 7` cells
  (20 of 144), are returned in `res.lookalikes` — centre, corners, wrong
  border and data cells, contrast — to steer towards the marker before its
  cells read.

Leave room for the height error and for tilt (the side shrinks by cos θ): ±30 %
is a reasonable window. A wrong height costs nothing in reading: every quad
is read the plain way first, so a marker that reads without the hint reads
with any hint, same corners. A far too small window only costs time (the
minimum contour drops to fit it: (5, 8) px doubles a frame without a marker,
0.57 → 1.17 ms on all cores) and still gave no false read.

```python
far = np.full((240, 320), 255, np.uint8)
far[60:130, 100:170] = nf.generate_fractal("FRACTAL_3L_6", 70)   # 5 px/cell
far[70:75, 110:160] ^= 255                 # ten cells of the outer grid read wrong
print(det3.detect(far).ids.tolist())       # plain: nothing

side = 0.18 * 1050 / 2.7                   # 0.18 m marker, f = 1050 px, 2.7 m: 70 px
res = det3.detect(far, expected_side_px=(side / 1.3, side * 1.3))
print(res.ids.tolist(), res.bit_errors.tolist())       # [0] [10]

worse = far.copy()
worse[75:80, 110:150] ^= 255               # eight more cells: 18 off
res = det3.detect(worse, expected_side_px=(side / 1.3, side * 1.3))
print(res.ids.tolist(), [(q.centre.round(), q.bit_errors) for q in res.lookalikes])
assert len(res) == 0 and res.lookalikes[0].bit_errors == 18
```

Flight videos (8027 frames, the window ±30 % around the marker's size,
tracker with `capture_frames=3`): 4818 frames with the marker against 4480,
the first read 9–36 s earlier on five approaches, lookalikes on 92 more
frames. Where there is no marker
(611 flight and synthetic frames, every window from 20 to 96 px) no
detection and no lookalike; the closest quad of the expected size there is
25+ cells off the code, the marker on the approach 9–24. Cost: +0.2–0.3 ms
per frame on one core.

## A camera loop

```python
# requires: cv2 and a camera
import time

import cv2
import nanofractal as nf

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.20)
tracker = nf.FractalTracker(det)
smoother = nf.PoseSmoother(process_noise=1e-3, measurement_noise=1e-4, gate=4.7)
K, D = ..., ...                      # your calibration

while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = time.monotonic()                             # frame time, seconds
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # or pass `frame` (BGR) directly
    res = tracker.detect(gray, with_inner_points=True)
    pose = det.estimate_pose(res, K, D)
    if pose is None:
        smoother.reset()
    elif pose[2] < 2.0:                              # gate on reprojection error
        rvec, tvec = smoother.update(pose[0], pose[1], t=t, reproj_err=pose[2],
                                     n_points=len(res.points_2d))
        det.draw(frame, res, K, D, rvec, tvec)
    cv2.imshow("nanofractal", frame)
    if cv2.waitKey(1) == 27:
        break
```

If the camera can deliver grayscale (many machine-vision cameras, or the Y
plane of a YUV stream), pass it directly and skip the conversion.

## Several cameras or streams

`detect()` uses all cores for one frame. With several streams processed at
the same time that would make them compete, so give each stream its own
detector and let each `detect()` use one core:

```python
import threading
import numpy as np
import nanofractal as nf

nf.set_num_threads(1)          # process-wide: one core per detect() call

def worker(frames, out):
    det = nf.FractalDetector("FRACTAL_5L_6")    # one detector per thread
    tracker = nf.FractalTracker(det)
    out.extend(len(tracker.detect(f)) for f in frames)

frame = np.full((720, 1280), 255, np.uint8)
frame[200:500, 300:600] = nf.generate_fractal("FRACTAL_5L_6", 300)
outs = [[] for _ in range(4)]
threads = [threading.Thread(target=worker, args=([frame] * 10, o)) for o in outs]
for t in threads:
    t.start()
for t in threads:
    t.join()
nf.set_num_threads(0)          # back to all cores
print([sum(o) for o in outs])
```

Detection releases the GIL, so threads scale across cores; separate
processes work too. For recorded footage, `detect_batch` does the same over a
list of frames in one call.
