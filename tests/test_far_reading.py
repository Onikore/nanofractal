"""Second reading of rejected quads: a far marker, ~3 px per cell (R16)."""

import numpy as np

import nanofractal as nf
import nanofractal._nanofractal as _nf
from nanofractal.bench import _blur, _warp_onto

CONFIG = "FRACTAL_4L_6"
CELLS = 15  # outer grid of FRACTAL_4L_6, border included
W, H = 640, 360
SS = 6  # supersampling of the marker render


def _far_frame(rng, px_per_cell):
    m = nf.generate_fractal(CONFIG, 600).astype(np.float64)
    q = 50
    mm = np.full((600 + 2 * q,) * 2, 255.0)
    mm[q:-q, q:-q] = m
    half = px_per_cell * CELLS / 2 * mm.shape[0] / 600
    a = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64) * half
    quad = (base * (1 + rng.uniform(-0.1, 0.1, (4, 1)))) @ rot.T + [W / 2, H / 2]
    quad += rng.uniform(-5, 5, 2)
    x0, y0 = np.floor(quad.min(0)).astype(int) - 2
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 2
    sub = np.zeros(((y1 - y0) * SS, (x1 - x0) * SS))
    cov = np.zeros_like(sub)
    _warp_onto(sub, mm, (quad - [x0, y0]) * SS)
    _warp_onto(cov, np.ones_like(mm), (quad - [x0, y0]) * SS)
    sub = sub.reshape(y1 - y0, SS, x1 - x0, SS).mean((1, 3))
    cov = cov.reshape(y1 - y0, SS, x1 - x0, SS).mean((1, 3))
    img = np.full((H, W), 180.0)
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - cov) + sub
    img = _blur(img, 0.6) + rng.normal(0, 3, img.shape)
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


def test_far_marker_read():
    rng = np.random.default_rng(3)
    det = nf.FractalDetector(CONFIG)
    frames = [_far_frame(rng, 3.0) for _ in range(20)]
    found = sum(0 in det.detect(f).ids.tolist() for f in frames)
    ref = sum(
        0 in _nf._reference_detect(CONFIG, -1.0, nf.DetectorParams(), f, False)[0].tolist()
        for f in frames
    )
    assert found >= 18  # the reference reader finds (almost) none of these
    assert found > ref + 10


def test_far_marker_corners():
    rng = np.random.default_rng(4)
    det = nf.FractalDetector(CONFIG)
    f = _far_frame(rng, 3.5)
    res = det.detect(f)
    assert 0 in res.ids.tolist()
    c = res.corners[res.ids.tolist().index(0)]
    side = np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)
    assert np.all(np.abs(side - 3.5 * CELLS) < 0.15 * 3.5 * CELLS)
