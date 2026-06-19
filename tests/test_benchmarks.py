import time

import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf


def _scene(width, height, marker_id=0, cell=40):
    """A single ArUco marker (8*cell px) placed near the top-left of a frame."""
    grid = np.asarray(_nf._aruco_marker_image8(0, marker_id))
    up = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    canvas = np.full((height, width), 255, dtype=np.uint8)
    h, w = up.shape
    canvas[20 : 20 + h, 20 : 20 + w] = up
    return np.ascontiguousarray(canvas)


@pytest.mark.benchmark(group="aruco-latency")
@pytest.mark.parametrize("size", [(640, 480), (1280, 720), (1920, 1080)])
def test_aruco_single_frame_latency(benchmark, size):
    w, h = size
    img = _scene(w, h)
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=1)
    res = benchmark(det.detect, img)
    assert 0 in res.ids.tolist()


def test_aruco_batch_throughput_scaling(capsys):
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=1)
    imgs = [_scene(1280, 720, marker_id=i % 50) for i in range(64)]

    def run(threads):
        t0 = time.perf_counter()
        det.detect_batch(imgs, num_threads=threads)
        return time.perf_counter() - t0

    run(1)  # warm up
    t1 = run(1)
    t4 = run(4)
    with capsys.disabled():
        print(
            f"\nbatch 64 frames @720p: 1 thread={t1 * 1e3:.1f}ms  "
            f"4 threads={t4 * 1e3:.1f}ms  speedup={t1 / t4:.2f}x"
        )
    # Non-flaky guard: more threads must not be meaningfully slower than one.
    assert t4 <= t1 * 1.2
