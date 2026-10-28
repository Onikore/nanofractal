# 8. Troubleshooting

## The marker is not detected

Work through these in order:

1. **Right configuration?** A `FRACTAL_4L_6` detector does not see a
   `FRACTAL_5L_6` print.
2. **Big enough?** The external marker needs ~45 px of side in the image
   (about 3 px per cell, after `detection_scale`). Move closer, raise the
   resolution or remove `detection_scale`.
3. **Quiet zone.** The black border must be surrounded by white (≥ 1 cell).
   A marker printed edge-to-edge on a dark table, or touching the image
   border, is not found.
4. **Mirrored image?** A mirrored marker reads as a different code; some
   front cameras and screen grabs mirror the image — flip it back.
5. **Contrast.** Glare, very dim light or grey print: try `adaptive_c = 4`.
6. **Distortion.** Wide-angle lenses bend the edges; detect on the raw frame
   and try `approx_poly_rate = 0.07` (or undistort first).
7. **Blur.** Motion blur breaks the cell pattern; shorten the exposure.
8. **Very close?** Cells of 60 px and more are far larger than the automatic
   threshold block (11 px at 1280 wide); on a dark frame the contours then
   fall apart. `params.close_pass` (on by default) retries such frames on a
   shrunken copy; if it is off, or the frame is huge, set
   `adaptive_block_size` to about a cell (`31`–`61`).

Look at what the detector sees with the drawing helper:

```python
import numpy as np
import nanofractal as nf

det = nf.FractalDetector("FRACTAL_5L_6")
frame = np.full((600, 600), 255, np.uint8)
frame[100:500, 100:500] = nf.generate_fractal("FRACTAL_5L_6", 400)
res = det.detect(frame)
overlay = det.draw(frame, res, inplace=False)
print(res, overlay.shape)       # save `overlay` and inspect it
```

### Why a candidate was dropped

`diagnose` runs the same detection and lists every convex quad it looked at,
with the reason it was kept or dropped and how close it came to a real code:

```python
frame = np.full((600, 600), 255, np.uint8)
frame[100:500, 100:500] = nf.generate_fractal("FRACTAL_5L_6", 400)
frame[136:156, 186:206] ^= 255           # damage one cell of the outer level
d = det.diagnose(frame)
print(d)                                 # counts by reason
for q in d.quads:
    if q.reason in ("border", "bits") and q.bit_errors <= 3:
        print(q.nearest_id, q.bit_errors, q.contrast)   # nearly a marker
```

| `reason` | meaning |
|---|---|
| `"accepted"` | decoded and returned |
| `"duplicate"` | decoded, but a larger quad of the same level was kept (the same square traced twice, or a second copy of the marker) |
| `"border"` | some cells of the black frame read white |
| `"bits"` | the frame is black, but no code matches |
| `"inconsistent"` | decoded, but not where the other levels put it (dropped, see `inconsistent_ids`) |

A quad of the marker's size that misses its id by one to three bits
(`bit_errors`) is a reading problem: too few pixels per cell, blur, or a
shadow edge across it (low `contrast`). Quads with many wrong bits are other
squares in the scene. Contours shorter than `min_contour_size` are not
listed; `n_not_quad` counts the ones that are not a convex quad.

## The pose jumps or is wrong

- Check `reproj_err`: above ~2 px the correspondences or the intrinsics are
  off. Gate on it.
- Use `with_inner_points=True` — four corners of a small marker leave the
  rotation poorly constrained (the classic planar flip ambiguity).
- Make sure `marker_size` is the printed **black square** side, in the unit
  you expect the translation in.
- Wrong or missing distortion coefficients bend everything near the image
  edges; calibrate.
- Smooth with `PoseSmoother` and reset it when the marker is lost.

## Errors

| error | cause |
|---|---|
| `TypeError` in `detect` | not `uint8`, or not C-contiguous → `np.ascontiguousarray(img, dtype=np.uint8)` |
| `ValueError: ... shape` | 4-channel (BGRA) or 1-D input → convert to gray or BGR |
| `ValueError: invalid config` | typo in the configuration name |
| `ValueError: adaptive_block_size ...` (or another field) | parameter out of range, see [Tuning](06-tuning.md) |
| `ValueError: roi ...` | ROI outside the image or empty |
| `ValueError` in `estimate_pose` | `camera_matrix` not 3×3, unsupported number of distortion coefficients, NaN/inf values, or `fisheye=True` without exactly 4 coefficients |

## Performance is lower than expected

- Call `nf.get_num_threads()`: something may have set it to 1.
- Several streams at once? See [Video](04-video.md#several-cameras-or-streams).
- Running `python -m nanofractal.bench` shows what your machine does on the
  reference scenes; compare with [Performance](05-performance.md).
