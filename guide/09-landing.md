# Landing on a marker

A recipe for a drone that lands on a fractal marker: one function per frame
that says what to do — search, steer towards something that looks like the
marker, or use the pose. It puts together what the other pages explain one by
one: the tracker ([Video](04-video.md)), the size from a rangefinder, the
lookalikes, and the pose with its uncertainty ([Pose](03-pose.md)).

## One step per frame

```python
import numpy as np
import nanofractal as nf

F = 1000.0                                   # focal length, px (from your calibration)
K = np.array([[F, 0, 320], [0, F, 240], [0, 0, 1]])
DIST = np.zeros(5)
SIZE = 0.18                                  # printed side of the marker, m

det = nf.FractalDetector("FRACTAL_3L_6", marker_size=SIZE)
tracker = nf.FractalTracker(det, capture_frames=3)   # also reads through the last result


def landing_step(frame, height_m=None):
    """('pose', Pose) | ('steer', centre px) | ('search', None)"""
    hint = None
    if height_m:                             # a rangefinder: the marker's side in px
        side = SIZE * F / height_m
        hint = (side / 1.3, side * 1.3)      # room for the height error and for tilt
    res = tracker.detect(frame, expected_side_px=hint)
    if len(res):
        return "pose", det.estimate_pose(res, K, DIST, full=True)
    if res.lookalikes:                       # the marker's square, cells not readable yet
        return "steer", res.lookalikes[0].centre
    return "search", None
```

- `FractalTracker(capture_frames=3)` keeps the marker once read (it reads the
  next frames through the last result, so a few cells lost to blur do not
  drop it) and picks it up before its first clean read.
- `expected_side_px` from the height reads a far marker with a looser
  tolerance — on overexposed flight video the first read came 9–36 s
  earlier — and fills `res.lookalikes` when the cells still do not read.
- A wrong height does not lose a marker that reads without it; with no
  height, pass `None`.

## A simulated approach

Frames of the marker growing from 4 to 7 px per cell (the drone descending),
with cells read wrong on the far ones, as blur or glare would:

```python
def frame_with_marker(px_per_cell, wrong_cells=0):
    big = nf.generate_fractal("FRACTAL_3L_6", 14 * px_per_cell * 4).astype(float)
    for k in range(wrong_cells):             # flip outer-grid cells (row 2, then row 3)
        r, c = 2 + k // 10, 2 + k % 10
        cell = 4 * px_per_cell
        big[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] *= -1
        big[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] += 255
    side = 14 * px_per_cell
    small = big.reshape(side, 4, side, 4).mean((1, 3))      # 4x area downscale
    frame = np.full((480, 640), 200.0)
    frame[240 - side // 2:240 - side // 2 + side, 320 - side // 2:320 - side // 2 + side] = small
    return frame.astype(np.uint8)


approach = [  # (frame, height from the rangefinder)
    (frame_with_marker(4, wrong_cells=18), SIZE * F / 56),
    (frame_with_marker(4, wrong_cells=12), SIZE * F / 56),
    (frame_with_marker(5, wrong_cells=4), SIZE * F / 70),
    (frame_with_marker(6), SIZE * F / 84),
    (frame_with_marker(7), SIZE * F / 98),
]
steps = [landing_step(frame, h) for frame, h in approach]
for (what, x), (_, h) in zip(steps, approach):
    print(f"{h:.2f} m: {what}", x.tvec.round(3) if what == "pose" else x)
assert [s[0] for s in steps] == ["steer", "pose", "pose", "pose", "pose"]
```

At 3.2 m the marker is 18 cells off its code: no reading accepts it, but it
is a lookalike of the expected size, and its centre is enough to steer. One
frame later, 12 cells off, the height hint reads it; from then on the tracker
keeps it.

## When to trust the pose

`Pose.sd_t` is the position uncertainty in metres. Far away, only the outer
corners are seen and it varies a lot from frame to frame (about ×5; above,
7.2 → 0.0 → 5.7 mm — 0 when the four corners happen to fit exactly), so
decide on a running median, relative to the height:

```python
from collections import deque

recent = deque(maxlen=3)
for what, pose in steps:
    if what != "pose":
        recent.clear()                       # lost: start again
        continue
    recent.append(pose.sd_t)
    height = float(pose.tvec[2])
    trusted = len(recent) == recent.maxlen and np.median(recent) < 0.02 * height
    print(f"z {height:.2f} m  sd_t {pose.sd_t * 1000:.1f} mm  trusted: {trusted}")
```

Thresholds to start from (measure on your own flights): a running median of
`sd_t` under ~2 % of the height, `reproj_err_rel` under 0.005, and
`res.bit_errors` 0 on at least one level (a level read through the prior or
the height hint has `bit_errors > 0`). `PoseSmoother` ([Pose](03-pose.md))
smooths the pose itself.

## What it cannot do

- **A marker cut by the frame edge.** At touchdown, if the camera looks past
  the centre, every level can be cut and nothing is found — mount the camera
  or place the marker so the centre stays in view.
- **Below ~2.5 px per cell.** The cells cannot be told apart; a
  `FRACTAL_3L_6` print of 18 cm is about 45 px wide at 4 m with f ≈ 1000 px
  (3.2 px per cell) and 30 px at 6 m. For more height, print the marker
  bigger: the range grows with its size.
- **Glare.** A glossy print saturates: the white is lost and the black cells
  turn grey. A matte print, or a shorter exposure, reads from farther away
  than any setting of the detector.
