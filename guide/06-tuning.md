# 6. Tuning parameters

Defaults work for printed markers in normal lighting. Change one thing at a
time and check the effect on your own footage.

```python
import nanofractal as nf

p = nf.DetectorParams()
print(p)    # DetectorParams(min_contour_size=-1, adaptive_block_size=-1, ...)
p.adaptive_c = 5.0
det = nf.FractalDetector("FRACTAL_5L_6", params=p)
det.params.subpix_win_size = 3      # changes apply from the next detect()
```

Invalid values raise `ValueError` when `detect()` is called.

| field | default | what it does | when to change |
|---|---|---|---|
| `detection_scale` | `1.0` | runs threshold + contours + decoding on a downscaled frame; corners are refined at full resolution; ≥ 1 or ≤ 0 = off | ≥ 4K frames: `0.5`; any factor in (0, 1) works — see [Performance](05-performance.md) |
| `adaptive_block_size` | `-1` (auto: `15 · width / 1920`, odd, ≥ 3) | window of the local-mean threshold | strong uneven lighting or very small/large markers relative to the frame. Odd, ≥ 3 |
| `adaptive_c` | `7.0` | a pixel is "dark" when it is this much darker than its local mean | low contrast (dim light, grey print): lower to 4–5. Many false contours in texture: raise to 9–12 |
| `min_contour_size` | `-1` (120 px of perimeter) | shorter contours are ignored | rarely: the detection limit for small markers comes from reading their cells, not from this |
| `approx_poly_rate` | `0.05` | polygon-fit tolerance as a fraction of the contour length | strong lens distortion bends marker edges: try `0.07` |
| `subpix_win_size` | `-1` (4 → 9×9 window) | corner sub-pixel refinement half-window; `0` disables it | small markers (< ~60 px): `2`–`3`; very blurry frames: `5`–`6` |
| `close_pass` | `True` | when a frame yields nothing, threshold it again shrunk to ~320 px wide with a block of a fifth of that width: finds markers seen from very close (cells far larger than the automatic block), above all on dark frames. Frames where the first pass finds a level are untouched | `False` only to save the extra 0.1–0.5 ms on frames without a marker |
| `kfilter_min_dist` | `10.0` | minimum distance between inner-corner keypoints (inner-point path only) | dense small markers: lower to 5–7 |

Notes:

- `min_contour_size` is always expressed in full-resolution pixels, also
  with `detection_scale`.
- Parameters belong to one detector; `detection_scale` etc. can be changed
  between frames (`det.params.detection_scale = 0.5`).
