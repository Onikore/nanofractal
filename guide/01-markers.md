# 1. Markers

## Configurations

A configuration fixes how many levels are nested and their bit patterns:

| config | levels (ids) | external marker grid | smallest level, relative to the external side | crisp `size_px`: multiple of |
|---|---|---|---|---|
| `FRACTAL_2L_6` | 0, 1 | 10×10 bits | 1/3 | 24 |
| `FRACTAL_3L_6` | 0, 1, 2 | 12×12 bits | 1/7 | 56 |
| `FRACTAL_4L_6` | 0, 1, 2, 3 | 13×13 bits | 1/15 | 120 |
| `FRACTAL_5L_6` | 0, 1, 2, 3, 4 | 11×11 bits | 1/39 | 312 |

Id 0 is always the external (largest) marker; higher ids are nested further
inside. Pick by the **range of distances** you need:

- the external marker is detected while it is at least ~50–60 px wide in the
  image (measured: 44 px for `2L`, 48 px for `5L`, 60 px for `4L` on a clean
  print);
- the innermost level keeps the marker detectable when the camera is so close
  that only the centre of the marker is in view.

More levels → a wider distance range at the same detection cost, so
`FRACTAL_5L_6` is the default choice; the smaller configurations matter
mainly when markers of that kind are already printed.

The detector only looks for the configuration it was created with, and it
reports each level id at most once per frame (if the same level is seen twice,
the larger instance wins). So use **one marker per configuration** in view; to
track several objects, give them markers of different configurations and use
one detector per configuration.

## Generating a marker

```python
import nanofractal as nf

img = nf.generate_fractal("FRACTAL_5L_6", size_px=2496)  # uint8, values 0/255
print(img.shape, img.dtype)
```

- The image covers the marker's **outer black border exactly** — leave a white
  margin around it (a quiet zone of at least one cell, i.e. ~1/13 of the
  side) when you place it on a page.
- Choose `size_px` as a multiple of the number in the last column of the
  table for perfectly sharp cell edges: the innermost cells of `FRACTAL_5L_6`
  are 1/312 of the side, so 2496 px (= 8 × 312) prints crisply. Other sizes
  work but shift cell edges by up to one pixel.
- `nested=False` renders only the external level (useful for tests of the
  long-range case).

## Printing

```python
import numpy as np
import nanofractal as nf

dpi = 300
side_m = 0.20                                  # the black square will be 20 cm
size_px = int(round(side_m / 0.0254 * dpi))    # 2362 px at 300 dpi
marker = nf.generate_fractal("FRACTAL_5L_6", size_px=size_px)
quiet = size_px // 12
page = np.full((size_px + 2 * quiet,) * 2, 255, np.uint8)
page[quiet:quiet + size_px, quiet:quiet + size_px] = marker
# save `page` with any image library (e.g. cv2.imwrite) and print at 300 dpi,
# "actual size" / no scaling. Measure the printed black square afterwards and
# use that number as marker_size.
```

Matte paper avoids specular highlights; a rigid flat backing keeps the marker
planar, which the pose assumes.

## How far away can it be seen?

A marker of side `L` metres at distance `Z` appears `f · L / Z` pixels wide,
where `f` is the focal length in pixels (from calibration; roughly
`width / (2·tan(HFOV/2))`). With the ~50 px detection limit:

```python
import math

width_px, hfov_deg = 1920, 70.0
f = width_px / (2 * math.tan(math.radians(hfov_deg) / 2))   # ~1371 px
L = 0.20                                                    # marker side, m
print("external level up to", round(f * L / 50, 1), "m")    # ~5.5 m
# the innermost FRACTAL_5L_6 level is L/39 wide: it needs the camera within
print("innermost level from", round(f * (L / 39) / 50, 2), "m")
```

Higher resolution scales the range linearly — and costs time; see
[Performance](05-performance.md) for `detection_scale`.
