import threading

import numpy as np

import nanofractal as nf

CONFIG = "FRACTAL_5L_6"


def _render_nested(size=780, margin=80, shift=0):
    """Full nested fractal marker on a white canvas (optionally shifted)."""
    marker = nf.generate_fractal(CONFIG, size_px=size)
    canvas = np.full((size + 2 * margin, size + 2 * margin + 40), 255, dtype=np.uint8)
    canvas[margin : margin + size, margin + shift : margin + shift + size] = marker
    return np.ascontiguousarray(canvas)


def _assert_same(seq, par):
    assert len(par) == len(seq)
    for a, b in zip(seq, par):
        assert a.ids.tolist() == b.ids.tolist()
        np.testing.assert_allclose(a.corners, b.corners, atol=1e-4)


def test_fractal_batch_matches_sequential(render_fractal):
    det = nf.FractalDetector(CONFIG)
    imgs = [render_fractal()] * 8
    seq = [det.detect(im) for im in imgs]
    _assert_same(seq, det.detect_batch(imgs, num_threads=4))


def test_nested_batch_matches_sequential():
    det = nf.FractalDetector(CONFIG)
    imgs = [_render_nested(shift=s) for s in (0, 7, 13, 29)]
    seq = [det.detect(im) for im in imgs]
    assert all(len(r) >= 3 for r in seq)  # several nested levels per frame
    _assert_same(seq, det.detect_batch(imgs, num_threads=4))


def test_empty_batch_returns_empty_list():
    assert nf.FractalDetector(CONFIG).detect_batch([]) == []


def test_batch_single_thread_matches():
    det = nf.FractalDetector(CONFIG)
    imgs = [_render_nested(shift=s) for s in (0, 11)]
    _assert_same(det.detect_batch(imgs, num_threads=1), det.detect_batch(imgs, num_threads=4))


def test_batch_default_threads_all_cores():
    # Default num_threads=0 -> use all cores; results must match sequential.
    det = nf.FractalDetector(CONFIG)
    imgs = [_render_nested(shift=s) for s in (0, 5, 17, 23)]
    seq = [det.detect(im) for im in imgs]
    _assert_same(seq, det.detect_batch(imgs))


def test_batch_releases_gil():
    # A background Python thread must keep advancing while detect_batch runs,
    # proving the GIL is released during the C++ detection.
    det = nf.FractalDetector(CONFIG)
    imgs = [_render_nested(shift=i % 30) for i in range(32)]
    counter = {"n": 0}
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter["n"] += 1

    t = threading.Thread(target=spin)
    t.start()
    try:
        det.detect_batch(imgs, num_threads=4)
    finally:
        stop.set()
        t.join()
    # If the GIL were held for the whole call the spinner could not advance.
    assert counter["n"] > 0
