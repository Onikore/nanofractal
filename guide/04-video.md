# 4. Video and cameras

## FractalTracker

In video the marker moves little between frames. `FractalTracker` wraps a
detector and searches only the region around the previous frame's markers
(grown by `margin` × their size + `pad` pixels). It falls back to the whole
frame whenever nothing was found before, the region finds nothing, or it finds
fewer levels than the previous frame — so a marker is never lost just because
the box was tight. Results are the same as `detect()`; measured on a moving
marker: **~2× faster at 4K, 1.4–1.8× at 1080p**, with identical corner
accuracy.

```python
import numpy as np
import nanofractal as nf

det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.20)
tracker = nf.FractalTracker(det, margin=0.25, pad=16)

# simulated video: the marker drifts to the right
marker = nf.generate_fractal("FRACTAL_5L_6", 300)
for i in range(5):
    frame = np.full((720, 1280), 255, np.uint8)
    frame[200:500, 100 + 60 * i:400 + 60 * i] = marker
    res = tracker.detect(frame)            # same arguments as detector.detect
    print(i, res.ids.tolist())

print("whole-frame searches:", tracker.full_frame_detections)
tracker.reset()                            # e.g. after a scene cut
```

`tracker.detect(frame, with_inner_points=True)` works the same way for pose.

## A camera loop

```python
# requires: cv2 and a camera
import time

import cv2
import nanofractal as nf

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
det = nf.FractalDetector("FRACTAL_5L_6", marker_size=0.20)
tracker = nf.FractalTracker(det)
smoother = nf.PoseSmoother(process_noise=1e-3, measurement_noise=1e-4, gate=4.7)
K, D = ..., ...                      # your calibration

while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = time.monotonic()                             # frame time, seconds
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # or pass `frame` (BGR) directly
    res = tracker.detect(gray, with_inner_points=True)
    pose = det.estimate_pose(res, K, D)
    if pose is None:
        smoother.reset()
    elif pose[2] < 2.0:                              # gate on reprojection error
        rvec, tvec = smoother.update(pose[0], pose[1], t=t, reproj_err=pose[2],
                                     n_points=len(res.points_2d))
        det.draw(frame, res, K, D, rvec, tvec)
    cv2.imshow("nanofractal", frame)
    if cv2.waitKey(1) == 27:
        break
```

If the camera can deliver grayscale (many machine-vision cameras, or the Y
plane of a YUV stream), pass it directly and skip the conversion.

## Several cameras or streams

`detect()` uses all cores for one frame. With several streams processed at
the same time that would make them compete, so give each stream its own
detector and let each `detect()` use one core:

```python
import threading
import numpy as np
import nanofractal as nf

nf.set_num_threads(1)          # process-wide: one core per detect() call

def worker(frames, out):
    det = nf.FractalDetector("FRACTAL_5L_6")    # one detector per thread
    tracker = nf.FractalTracker(det)
    out.extend(len(tracker.detect(f)) for f in frames)

frame = np.full((720, 1280), 255, np.uint8)
frame[200:500, 300:600] = nf.generate_fractal("FRACTAL_5L_6", 300)
outs = [[] for _ in range(4)]
threads = [threading.Thread(target=worker, args=([frame] * 10, o)) for o in outs]
for t in threads:
    t.start()
for t in threads:
    t.join()
nf.set_num_threads(0)          # back to all cores
print([sum(o) for o in outs])
```

Detection releases the GIL, so threads scale across cores; separate
processes work too. For recorded footage, `detect_batch` does the same over a
list of frames in one call.
