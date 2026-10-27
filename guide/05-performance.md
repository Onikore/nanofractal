# 5. Performance: how to use it most efficiently

Numbers below: Intel i5-12400 (6 cores / 12 threads), published wheels
(OpenCV 4.10, AVX2), `FRACTAL_5L_6`, a marker in a cluttered scene, median
per frame. Measure your own machine with the commands at the end.

## Decision table

| situation | do this | why |
|---|---|---|
| live video, one camera | `FractalTracker(detector)`, default threads | searches near the last detection: ~2× at 4K; all cores inside each frame |
| 4K (or larger) frames, marker not tiny | `params.detection_scale = 0.5` | threshold/contours on a quarter of the pixels; corners still refined at full resolution |
| several cameras / streams at once | `nf.set_num_threads(1)`, one detector (or tracker) per stream | streams run in parallel instead of competing for cores |
| a folder of images / recorded video | `detector.detect_batch(frames)` | one frame per core, best throughput |
| marker position known roughly | `detect(image, roi=(x, y, w, h))` | only that region is processed |
| need pose | `with_inner_points=True` | costs ~1.5 ms extra at 1080p, much better pose |
| only need to know where the marker is | plain `detect()` | skip the inner-point stage |
| colour camera | pass gray if you have it; BGR is fine too | conversion is only ~0.07 ms at 1080p |

## Measured

| | 1080p | 4K |
|---|---:|---:|
| `detect()`, all cores | 0.9–1.1 ms | 3.7–4.2 ms |
| `detect()`, 1 core | 3.2 ms | 12.5 ms |
| `detect(with_inner_points=True)`, all cores | 2.3–2.5 ms | 7.4 ms |
| `FractalTracker`, all cores (moving marker) | 0.57 ms | 2.2 ms |
| `detection_scale=0.5`, all cores | 0.37 ms | 0.94 ms |
| `detect_batch`, 48 frames | ~1360 frames/s | — |

Version 0.5 needed 12–14 ms at 1080p and ~50 ms at 4K for the same frames.

## detection_scale: use 0.5 or 0.25

`detection_scale` runs the expensive stages (adaptive threshold, contour
tracing, decoding) on a downscaled copy, then refines the corners on the
full-resolution image, so accuracy stays sub-pixel as long as the markers stay
large enough after scaling.

**Only use 0.5 or 0.25.** OpenCV's area resize has a fast path for integer
reduction factors only; other factors are slower than not scaling at all:

| 4K frame | time | levels found |
|---|---:|---|
| 1.0 | 3.7 ms | 0, 1, 2, 3 |
| 0.75 | 4.3 ms | 0, 1, 2, 3 |
| **0.5** | **0.94 ms** | 0, 1, 2, 3 |
| 0.35 | 3.0 ms | 0, 1, 2 |

The price of scaling is range: a level must still be ~50 px wide *after*
scaling, so the smallest inner levels (and markers far away) drop out first.
Good fit: 4K cameras watching a marker that fills a decent part of the frame.

```python
import nanofractal as nf

p = nf.DetectorParams()
p.detection_scale = 0.5
det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.20, params=p)
```

## Threads

`nf.set_num_threads(n)` is process-wide and caps how many cores one
`detect()` uses. Speed-up of a 1080p frame on 6 physical cores:

| threads | 1 | 2 | 4 | 6 | 12 |
|---|---:|---:|---:|---:|---:|
| ms | 3.2 | 1.8 | 1.4 | 1.1 | 1.1–1.4 |

Gains stop around the number of **physical** cores. On a machine shared with
other heavy work, setting it to the physical-core count (or fewer) keeps
latency steady. `detect_batch` is not affected: it always runs one frame per
worker and never splits frames.

## Cheap things that matter

- **Create detectors once**, outside the frame loop (creation is cheap, but
  the first frames also warm up internal buffers that are then reused).
- Keep frames **C-contiguous `uint8`**; strided views raise rather than copy.
- Don't request inner points on frames where you don't use the pose.
- Gate poses on `reproj_err` instead of lowering detection thresholds.

## Measure on your machine

```bash
python -m nanofractal.bench --resolution 1920x1080 --scene perspective --mode detect
python -m nanofractal.bench --resolution 3840x2160 --mode scale05
```

`--scene clean|perspective|clutter`, `--mode detect|inner|scale05`. The
repository's `benchmarks/fractal_bench.py` runs a full matrix and also
reports `detect_batch` throughput.
