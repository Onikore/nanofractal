# nanofractal user guide

Fractal markers are square fiducial markers nested inside each other. The
outermost marker is seen from far away; as the camera gets closer and the
outer marker leaves the frame (or gets occluded), the inner markers are still
detected. Every level that is visible contributes corner correspondences, so
pose stays accurate at long and short range and under partial occlusion.

`nanofractal` detects them from NumPy images, fast: about **1 ms per 1080p
frame and 4 ms per 4K frame** on a 6-core desktop CPU.

| page | what you'll find |
|---|---|
| [1. Markers](01-markers.md) | choosing a configuration, generating and printing markers, detection range |
| [2. Detection](02-detection.md) | inputs, results, levels and corner order, regions of interest |
| [3. Pose](03-pose.md) | camera calibration, 6-DoF pose, inner corners, fisheye, drawing, smoothing |
| [4. Video and cameras](04-video.md) | `FractalTracker`, a camera loop, several cameras |
| [5. Performance](05-performance.md) | **how to use it most efficiently** — decision table and measured numbers |
| [6. Tuning parameters](06-tuning.md) | every `DetectorParams` field and when to change it |
| [7. API reference](07-api.md) | all public classes and functions |
| [8. Troubleshooting](08-troubleshooting.md) | marker not detected, noisy pose, common errors |

## Install

```bash
pip install nanofractal
```

Wheels for Linux x86_64 / aarch64 and Windows x64 (Python 3.9–3.13) bundle
OpenCV; nothing else is required. x86_64 wheels need an AVX2 CPU (2013+).

## Sixty seconds

```python
import numpy as np
import nanofractal as nf

# a printable marker (here simulated: pasted on a white page)
marker = nf.generate_fractal("FRACTAL_5L_6", size_px=600)
page = np.full((800, 800), 255, np.uint8)
page[100:700, 100:700] = marker

det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.20)  # printed side: 20 cm
res = det.detect(page)
print(res)                    # DetectionResult(n=4, ids=[...])
for level, corners in res:    # one entry per visible nested level
    print(level, corners.shape)

# pose with a (here: guessed) camera
K = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]], np.float64)
rvec, tvec, err = det.estimate_pose(det.detect(page, with_inner_points=True), K, np.zeros(5))
print("distance (m):", float(np.linalg.norm(tvec)), "reprojection (px):", err)
```

Every Python snippet in this guide is executed by the test-suite
(`tests/test_guide.py`), so the examples match the library you installed.
