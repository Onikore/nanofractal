#!/usr/bin/env python3
"""Batch process multiple images for ArUco marker detection.

This script detects markers in multiple images using the parallel batch API.
Reads images from a folder (provided as command line argument) or generates
synthetic frames.

Usage:
    python examples/batch_processing.py [image_folder]
"""

import os
import sys
from pathlib import Path

import numpy as np

import nanofractal as nf


def load_images_from_folder(folder_path):
    """Load all .png, .jpg, .jpeg images from a folder."""
    try:
        import cv2
    except ImportError:
        print("Warning: opencv-python not available; cannot load images from folder.")
        return None

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(Path(folder_path).glob(ext))

    if not image_files:
        print(f"No images found in {folder_path}")
        return None

    images = []
    for img_path in sorted(image_files):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            print(f"  Loaded {img_path.name}")

    return images if images else None


def generate_synthetic_frames(num_frames=5):
    """Generate synthetic frames with ArUco markers."""
    print(f"Generating {num_frames} synthetic frames with ArUco markers...")
    frames = []
    for i in range(num_frames):
        # Generate a marker with id = i % 10
        marker_id = i % 10
        marker = nf.generate_aruco(marker_id, size_px=200, dictionary=nf.Dict.DICT_4X4_50)

        # Create a white canvas
        canvas_size = 400
        frame = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

        # Place marker at a varying offset to simulate different frames
        offset_x = (i * 20) % (canvas_size - 200)
        offset_y = (i * 15) % (canvas_size - 200)
        frame[offset_y : offset_y + 200, offset_x : offset_x + 200] = marker

        # Convert to BGR for consistency
        frame_bgr = np.stack([frame, frame, frame], axis=2)
        frames.append(frame_bgr)
        print(f"  Frame {i}: marker id {marker_id}")

    return frames


def main():
    folder_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Load or generate images
    if folder_path and os.path.isdir(folder_path):
        print(f"Loading images from {folder_path}...")
        images = load_images_from_folder(folder_path)
        if images is None:
            print("Falling back to synthetic frames...")
            images = generate_synthetic_frames(5)
    else:
        if folder_path:
            print(f"Folder not found: {folder_path}")
        print("Generating synthetic frames...")
        images = generate_synthetic_frames(5)

    if not images:
        print("Error: No images to process")
        sys.exit(1)

    print(f"\nProcessing {len(images)} images...")

    # Detect markers using batch API
    detector = nf.ArucoDetector(nf.Dict.DICT_4X4_50)

    # Use num_threads=0 for automatic (all cores)
    print("Running batch detection with num_threads=0 (all cores)...")
    results = detector.detect_batch(images, num_threads=0)

    # Print results
    print("\nBatch detection results:")
    total_markers = 0
    for frame_idx, result in enumerate(results):
        print(f"  Frame {frame_idx}: detected {len(result)} marker(s)", end="")
        if len(result) > 0:
            print(f" - IDs: {result.ids.tolist()}")
            total_markers += len(result)
        else:
            print()

    print(f"\nTotal markers detected: {total_markers}")


if __name__ == "__main__":
    main()
