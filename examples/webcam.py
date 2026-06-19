#!/usr/bin/env python3
"""Real-time ArUco marker detection from webcam.

This script captures video from the default webcam and detects ArUco markers
in real-time. Draws the detected markers and their IDs on each frame.

**Requires opencv-python:** `pip install opencv-python`

Usage:
    python examples/webcam.py

Controls:
    - Press 'q' to quit
    - Press 's' to save a frame with detections
"""

import sys


def main():
    # Check for opencv-python availability first
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is required for this example.")
        print("Install it with: pip install opencv-python")
        sys.exit(1)

    import nanofractal as nf

    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Creating detector...")
    detector = nf.ArucoDetector(nf.Dict.DICT_4X4_50)

    print("Starting video stream (press 'q' to quit, 's' to save)...")
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam")
            break

        frame_count += 1

        # Detect markers
        result = detector.detect(frame)

        # Draw detections
        annotated = detector.draw(frame, result, inplace=False)

        # Add frame info
        info = f"Frame: {frame_count} | Markers: {len(result)}"
        cv2.putText(
            annotated,
            info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Display
        cv2.imshow("nanofractal - ArUco Detection", annotated)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("s"):
            saved_count += 1
            output_path = f"webcam_frame_{saved_count:03d}.png"
            cv2.imwrite(output_path, annotated)
            print(f"Saved frame to {output_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
