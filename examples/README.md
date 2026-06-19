# Examples

This directory contains runnable example scripts demonstrating the nanofractal API.

## Available Examples

### `detect_image.py`
Detect ArUco markers in a single image.

- Loads an image from the command line argument, or generates a fallback marker if not provided.
- Detects markers using `ArucoDetector.detect()`.
- Draws the detected marker outlines and IDs.
- Saves the result to `detected.png` (or `detected.pgm` if opencv-python is not installed).

**Usage:**
```bash
python examples/detect_image.py [image_path]
```

### `pose_estimation.py`
Estimate the 3D pose of detected ArUco markers.

- Detects markers and estimates their 6DoF pose using `ArucoDetector.estimate_pose()`.
- Prints rotation vectors (rvec), translation vectors (tvec), and reprojection errors.
- Draws the detected markers and pose axes if opencv-python is available.
- Saves the visualization to `pose_estimation.png`.

**Usage:**
```bash
python examples/pose_estimation.py [image_path]
```

### `batch_processing.py`
Process multiple images using the parallel batch detection API.

- Loads images from a folder (command line argument) or generates synthetic frames.
- Detects markers in all images in parallel using `ArucoDetector.detect_batch(num_threads=0)`.
- Prints detection results per frame.

**Usage:**
```bash
python examples/batch_processing.py [image_folder]
```

### `webcam.py`
Real-time marker detection from a webcam stream.

- Captures video from the default webcam.
- Detects and visualizes ArUco markers in real-time.
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
# Generate a marker and detect it
env -u PYTHONPATH .venv/bin/python examples/detect_image.py

# Estimate pose with a generated marker
env -u PYTHONPATH .venv/bin/python examples/pose_estimation.py

# Batch process synthetic frames
env -u PYTHONPATH .venv/bin/python examples/batch_processing.py

# Live webcam (requires opencv-python)
env -u PYTHONPATH .venv/bin/python examples/webcam.py
```

> **Note:** The `env -u PYTHONPATH` prefix is required in the nanofractal development environment to avoid conflicts with ROS paths. Omit it if not using ROS.
