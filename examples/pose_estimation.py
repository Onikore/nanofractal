#!/usr/bin/env python3
"""Estimate the pose of a fractal marker from an image.

Detects the marker with inner-corner correspondences (occlusion-robust, many
points) and solves its 6-DoF pose in the camera frame, reporting the RMS
reprojection error. With opencv-python installed the annotated image is saved.

Usage:
    python examples/pose_estimation.py [image_path] [--config FRACTAL_5L_6] [--size 0.2]
"""

import argparse

import numpy as np

import nanofractal as nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--config", default="FRACTAL_5L_6")
    ap.add_argument("--size", type=float, default=0.2, help="marker side in metres")
    args = ap.parse_args()

    image = None
    if args.image:
        try:
            import cv2

            image = cv2.imread(args.image)
            if image is None:
                raise SystemExit(f"Error: could not load image from {args.image}")
        except ImportError:
            print("Warning: opencv-python not available; generating a fallback marker.")

    if image is None:
        print(f"Generating a fallback {args.config} marker...")
        image = np.full((900, 900), 255, dtype=np.uint8)
        image[100:800, 100:800] = nf.generate_fractal(args.config, size_px=700)

    img_bgr = np.stack([image] * 3, axis=2) if image.ndim == 2 else image.copy()
    H, W = img_bgr.shape[:2]
    camera_matrix = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros(5, dtype=np.float64)

    detector = nf.FractalDetector(args.config, marker_size=args.size)
    result = detector.detect(img_bgr, with_inner_points=True)
    print(f"Detected levels {result.ids.tolist()}, {len(result.points_2d)} inner points")

    pose = detector.estimate_pose(result, camera_matrix, dist_coeffs)
    if pose is None:
        print("No marker detected.")
        return
    rvec, tvec, err = pose
    print(f"rvec: {rvec}\ntvec: {tvec}\nreproj_error: {err:.4f} px")

    annotated = detector.draw(
        img_bgr, result, camera_matrix, dist_coeffs, rvec, tvec, inplace=False
    )
    try:
        import cv2

        cv2.imwrite("pose_estimation.png", annotated)
        print("Saved pose visualization to pose_estimation.png")
    except ImportError:
        print("opencv-python not available; not saving the annotated image")


if __name__ == "__main__":
    main()
