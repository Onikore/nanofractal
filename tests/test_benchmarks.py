import time

import pytest

from nanofractal import bench


@pytest.mark.benchmark(group="fractal-latency")
@pytest.mark.parametrize("size", [(640, 480), (1280, 720), (1920, 1080)])
def test_fractal_single_frame_latency(benchmark, size):
    w, h = size
    frame = bench.generate_scene(w, h, "perspective")
    det = bench.make_detector("detect")
    res = benchmark(det.detect, frame)
    assert 0 in res.ids.tolist()


def test_fractal_batch_throughput_scaling(capsys):
    det = bench.make_detector("detect")
    imgs = [bench.generate_scene(1280, 720, "perspective", seed=i) for i in range(32)]

    def run(threads):
        t0 = time.perf_counter()
        det.detect_batch(imgs, num_threads=threads)
        return time.perf_counter() - t0

    run(1)  # warm up
    t1 = run(1)
    t4 = run(4)
    with capsys.disabled():
        print(
            f"\nbatch 32 frames @720p: 1 thread={t1 * 1e3:.1f}ms  "
            f"4 threads={t4 * 1e3:.1f}ms  speedup={t1 / t4:.2f}x"
        )
    # Non-flaky guard: more threads must not be meaningfully slower than one.
    assert t4 <= t1 * 1.2
