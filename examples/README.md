# Examples

This directory contains runnable example scripts demonstrating the nanofractal API.

## Available Examples

### `detect_image.py`
Detect fractal markers in a single image.

- Loads an image from the command line argument, or generates a fallback marker if not provided.
- Detects every visible marker level using `FractalDetector.detect()`.
- Draws the detected marker outlines and IDs.
- Saves the result to `detected.png` (or `detected.pgm` if opencv-python is not installed).

**Usage:**
```bash
python examples/detect_image.py [image_path]
```

### `pose_estimation.py`
Estimate the 3D pose of a fractal marker.

- Detects the marker with inner-corner correspondences (`with_inner_points=True`) and
  estimates its 6DoF pose using `FractalDetector.estimate_pose()`.
- Prints the rotation vector (rvec), translation vector (tvec), and reprojection error.
- Draws the detected markers and pose axes if opencv-python is available.
- Saves the visualization to `pose_estimation.png`.

**Usage:**
```bash
python examples/pose_estimation.py [image_path] [--config FRACTAL_5L_6] [--size 0.2]
```

### `batch_processing.py`
Process multiple images using the parallel batch detection API.

- Loads images from a folder (command line argument) or generates synthetic frames.
- Detects markers in all images in parallel using `FractalDetector.detect_batch(num_threads=0)`.
- Prints detection results per frame.

**Usage:**
```bash
python examples/batch_processing.py [image_folder]
```

### `webcam.py`
Real-time marker detection from a webcam stream.

- Captures video from the default webcam.
- Detects and visualizes fractal markers in real-time with `FractalTracker`
  (searches near the previous frame's markers).
- Press 'q' to quit, 's' to save a frame.

**Requires opencv-python:**
```bash
pip install opencv-python
python examples/webcam.py
```

## Dependencies

All examples require:
- **numpy** — array operations
- **nanofractal** — marker detection library

Most examples work with just numpy; opencv-python is only required for:
- `webcam.py` — **required** for video capture
- `detect_image.py` — optional, for loading images from disk
- `pose_estimation.py` — optional, for image loading and visualization
- `batch_processing.py` — optional, for loading images from disk

To use all features across all examples:
```bash
pip install opencv-python
```

## Running Examples

All examples are designed to work independently and generate fallback data if inputs are missing:

```bash
python examples/detect_image.py        # generate a marker and detect it
python examples/pose_estimation.py     # estimate pose with a generated marker
python examples/batch_processing.py    # batch process synthetic frames
python examples/webcam.py              # live webcam (requires opencv-python)
```

> **Note:** in a shell that sources a ROS workspace, run Python with
> `env -u PYTHONPATH` (see `DEV.md`).

For video from a moving camera (a drone landing, say), see
[guide/04-video.md](../guide/04-video.md): `FractalTracker(..., auto_scale=True,
use_prior=True)` and the reading quality of every level (`res.cell_px`,
`res.contrast`, `res.bit_errors`).
