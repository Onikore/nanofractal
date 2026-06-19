import threading

import numpy as np

import nanofractal as nf
import nanofractal._nanofractal as _nf


def _render_aruco(marker_id=0, cell=40, margin=60):
    grid = np.asarray(_nf._aruco_marker_image8(0, marker_id))
    up = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    h, w = up.shape
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = up
    return np.ascontiguousarray(canvas)


def _render_fractal(cell=40, margin=80):
    grid = np.asarray(_nf._fractal_external_image8("FRACTAL_5L_6"))
    up = np.kron(grid, np.ones((cell, cell), dtype=np.uint8))
    h, w = up.shape
    canvas = np.full((h + 2 * margin, w + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = up
    return np.ascontiguousarray(canvas)


def test_aruco_batch_matches_sequential():
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    imgs = [_render_aruco(i) for i in [0, 1, 7, 42]]
    seq = [det.detect(im) for im in imgs]
    par = det.detect_batch(imgs, num_threads=4)
    assert len(par) == len(seq)
    for a, b in zip(seq, par):
        assert a.ids.tolist() == b.ids.tolist()
        np.testing.assert_allclose(a.corners, b.corners, atol=1e-4)


def test_fractal_batch_matches_sequential():
    det = nf.FractalDetector("FRACTAL_5L_6")
    img = _render_fractal()
    imgs = [img] * 8
    seq = [det.detect(im) for im in imgs]
    par = det.detect_batch(imgs, num_threads=4)
    assert len(par) == len(seq)
    for a, b in zip(seq, par):
        assert a.ids.tolist() == b.ids.tolist()
        np.testing.assert_allclose(a.corners, b.corners, atol=1e-4)


def test_empty_batch_returns_empty_list():
    assert nf.ArucoDetector().detect_batch([]) == []
    assert nf.FractalDetector("FRACTAL_5L_6").detect_batch([]) == []


def test_batch_single_thread_matches():
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    imgs = [_render_aruco(i) for i in [0, 7]]
    one = det.detect_batch(imgs, num_threads=1)
    four = det.detect_batch(imgs, num_threads=4)
    for a, b in zip(one, four):
        assert a.ids.tolist() == b.ids.tolist()
        np.testing.assert_allclose(a.corners, b.corners, atol=1e-4)


def test_batch_default_threads_all_cores():
    # Default num_threads=0 -> use all cores; results must match sequential.
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    imgs = [_render_aruco(i) for i in [0, 1, 7, 42]]
    seq = [det.detect(im) for im in imgs]
    par = det.detect_batch(imgs)  # default num_threads=0
    assert len(par) == len(seq)
    for a, b in zip(seq, par):
        assert a.ids.tolist() == b.ids.tolist()
        np.testing.assert_allclose(a.corners, b.corners, atol=1e-4)


def test_batch_releases_gil():
    # A background Python thread must keep advancing while detect_batch runs,
    # proving the GIL is released during the C++ detection.
    det = nf.ArucoDetector(nf.Dict.ARUCO_MIP_36h12, max_attempts=10)
    imgs = [_render_aruco(i % 50, cell=60) for i in range(64)]
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
