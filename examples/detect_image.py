#!/usr/bin/env python3
"""Detect ArUco markers in an image.

This script loads an image (from command line or generates a fallback marker),
detects ArUco markers using nanofractal, and saves the annotated result.

If opencv-python is available, uses it to load/save images; otherwise generates
a marker and demonstrates detection with a simple PGM save.

Usage:
    python examples/detect_image.py [image_path]
"""

import sys

import numpy as np

import nanofractal as nf


def save_pgm(filename, image):
    """Save a uint8 grayscale image as PGM (simple, no dependencies)."""
    if image.ndim == 3:
        # Convert RGB/BGR to grayscale
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    with open(filename, "wb") as f:
        f.write(b"P5\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())
        f.write(b"255\n")
        f.write(image.tobytes())


def main():
    # Check if an image path was provided
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Try to load the image
    image = None
    if image_path:
        try:
            import cv2

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                sys.exit(1)
        except ImportError:
            print(
                "Warning: opencv-python not available; cannot load image. "
                "Generating a fallback marker."
            )

    # If no image loaded, generate a fallback marker
    if image is None:
        print("Generating a fallback ArUco marker (id=0, DICT_4X4_50)...")
        marker = nf.generate_aruco(0, size_px=300, dictionary=nf.Dict.DICT_4X4_50)
        # Create a white canvas and place the marker in the center
        canvas_size = 500
        image = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
        offset = (canvas_size - 300) // 2
        image[offset : offset + 300, offset : offset + 300] = marker

    # Detect markers
    print(f"Detecting markers in image shape {image.shape}...")
    detector = nf.ArucoDetector(nf.Dict.DICT_4X4_50)
    result = detector.detect(image)

    # Print results
    print(f"Detected {len(result)} marker(s)")
    if len(result) > 0:
        print(f"  IDs: {result.ids.tolist()}")
        print(f"  Corners shape: {result.corners.shape}")
        for mid, corners in result:
            print(f"    ID {mid}: corners shape {corners.shape}")

    # Draw detections
    print("Drawing detections...")
    try:
        # Try to convert to BGR if we have a grayscale image
        if image.ndim == 2:
            img_bgr = np.stack([image, image, image], axis=2)
        else:
            img_bgr = image.copy()

        annotated = detector.draw(img_bgr, result, inplace=False)

        # Save the result
        try:
            import cv2

            output_path = "detected.png"
            cv2.imwrite(output_path, annotated)
            print(f"Saved annotated image to {output_path}")
        except ImportError:
            output_path = "detected.pgm"
            save_pgm(output_path, annotated)
            print(f"Saved annotated image to {output_path} (PGM format, no cv2)")

    except Exception as e:
        print(f"Warning: Could not save annotated image: {e}")


if __name__ == "__main__":
    main()
