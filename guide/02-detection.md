# 2. Detection

## Input images

`detect()` takes a NumPy `uint8` array, **C-contiguous**:

- `(H, W)` or `(H, W, 1)` grayscale — the fastest input;
- `(H, W, 3)` BGR (OpenCV's order) — converted to gray internally.

The array is wrapped without copying. Wrong dtype or a non-contiguous view
(a slice with a step, a transposed array, ...) raises `TypeError` instead of
being copied behind your back — use `np.ascontiguousarray` once if needed.

```python
import numpy as np
import nanofractal as nf

det = nf.FractalDetector("FRACTAL_5L_6")
page = np.full((900, 900), 255, np.uint8)
page[100:800, 100:800] = nf.generate_fractal("FRACTAL_5L_6", 700)

res = det.detect(page)                       # grayscale
bgr = np.ascontiguousarray(np.dstack([page] * 3))
assert det.detect(bgr).ids.tolist() == res.ids.tolist()

view = page[:, ::2]                          # strided view
try:
    det.detect(view)
except TypeError:
    res2 = det.detect(np.ascontiguousarray(view))
```

## The result

`DetectionResult` holds one entry per **visible level** of the marker:

```python
res = det.detect(page)
print(len(res), bool(res))      # number of levels found; False if none
print(res.ids)                  # int32 (N,)      level ids, 0 = external
print(res.corners.shape)        # float32 (N,4,2) sub-pixel corners, (x, y) pixels
for level, corners in res:
    print(level, corners[0])    # first corner of each level
```

Corners are listed in a fixed order relative to the **marker**, not the image:
the marker's top-left, top-right, bottom-right, bottom-left (as printed,
looking at it). That order is stable under any in-plane rotation, so
`corners[0]` always identifies the same physical corner. Pixel coordinates use
OpenCV's convention: the centre of pixel `(i, j)` is at `x = j, y = i`.

`res.points_2d` / `res.points_3d` stay `None` unless you ask for inner corners
(see [Pose](03-pose.md)).

## Regions of interest

When you know where the marker can be, restrict the search. The ROI is a
zero-copy view; corners come back in **full-image** coordinates:

```python
res = det.detect(page, roi=(50, 50, 800, 800))   # (x, y, width, height)
print(res.corners[0][0])                          # full-image pixels
```

For video, `FractalTracker` does this automatically from the previous frame
([Video](04-video.md)).

## Many images at once

```python
frames = [page] * 8
results = det.detect_batch(frames)                  # all cores, one frame per core
results = det.detect_batch(frames, num_threads=4)   # explicit
print([len(r) for r in results])
```

`detect_batch` releases the GIL and runs one frame per worker thread; every
frame gets exactly the result `detect()` would give. Use it for offline
processing; for a live stream prefer `detect()` on each frame, which already
uses all cores *within* the frame.

## Refining corners yourself

`detect()` already returns sub-pixel corners. `refine_corners` re-runs the
refinement with a different window, e.g. on a sharper frame of the same
scene:

```python
refined = nf.refine_corners(page, res.corners, win_size=5)
print(refined.shape, refined.dtype)   # (N, 4, 2) float32
```

## Threads and the GIL

- The GIL is released during detection: other Python threads keep running.
- A `FractalDetector` reuses internal buffers, so **one detector per thread**
  if you call it from several threads at once (creating one costs ~10 µs).
- `nf.set_num_threads(n)` sets how many cores one `detect()` may use
  (default: all). See [Performance](05-performance.md).
