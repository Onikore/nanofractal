#!/usr/bin/env python3
"""Estimate ArUco marker pose from an image.

This script detects ArUco markers and estimates their 3D pose in the camera frame,
including reprojection error. If opencv-python is available, also draws pose axes.

Usage:
    python examples/pose_estimation.py [image_path]
"""

import sys

import numpy as np

import nanofractal as nf


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

    # Convert to BGR if grayscale
    if image.ndim == 2:
        img_bgr = np.stack([image, image, image], axis=2)
    else:
        img_bgr = image.copy()

    H, W = img_bgr.shape[:2]

    # Create a simple camera matrix (assuming ~50 degree FOV for VGA)
    focal_length = W
    camera_matrix = np.array(
        [[focal_length, 0, W / 2], [0, focal_length, H / 2], [0, 0, 1]], dtype=np.float64
    )
    # No distortion for simplicity
    dist_coeffs = np.zeros(4, dtype=np.float64)

    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Image shape: {img_bgr.shape}")

    # Detect markers
    print("Detecting markers...")
    detector = nf.ArucoDetector(nf.Dict.DICT_4X4_50)
    result = detector.detect(img_bgr)

    print(f"Detected {len(result)} marker(s)")

    if len(result) == 0:
        print("No markers detected.")
        return

    # Estimate pose with reprojection error
    marker_size = 0.05  # 5 cm
    print(f"\nEstimating pose (marker_size={marker_size} m)...")
    rvecs, tvecs, reproj_errs = detector.estimate_pose(
        result.corners, camera_matrix, dist_coeffs, marker_size, return_reproj=True
    )

    print("Pose estimates:")
    for i, (mid, corners) in enumerate(result):
        print(f"  ID {mid}:")
        print(f"    rvec: {rvecs[i]}")
        print(f"    tvec: {tvecs[i]}")
        print(f"    reproj_error: {reproj_errs[i]:.4f} px")

    # Draw detections and axes if cv2 is available
    try:
        annotated = detector.draw(
            img_bgr,
            result,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvecs=rvecs,
            tvecs=tvecs,
            marker_size=marker_size,
            axis_length=marker_size * 0.5,
            inplace=False,
        )

        try:
            import cv2

            output_path = "pose_estimation.png"
            cv2.imwrite(output_path, annotated)
            print(f"\nSaved pose visualization to {output_path}")
        except ImportError:
            print("\nWarning: opencv-python not available for saving annotated image")

    except Exception as e:
        print(f"Warning: Could not draw pose axes: {e}")


if __name__ == "__main__":
    main()
