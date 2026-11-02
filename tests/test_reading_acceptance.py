"""Acceptance floors on the synthetic scenarios of benchmarks/reading.py (R9).

Shadow edges, partial occlusion and marker-like negatives, the same frames as
the benchmark (first frames of its seeded sequences). 0.7.2 finds 20, 18 and
10 of 20 (as the reference detector: these frames test the threshold and the
contours, not the bit reading — that is test_far_reading); the floors sit one
frame below, so a lost frame is noise but two fail (adaptive_c=3 finds 16 under
the x0.45 shadow). Negatives must stay clean.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import nanofractal as nf

_SPEC = importlib.util.spec_from_file_location(
    "reading", Path(__file__).parent.parent / "benchmarks" / "reading.py"
)
reading = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reading)

FRAMES = 20


@pytest.fixture(scope="module")
def det():
    return nf.FractalDetector(reading.CONFIG)


def _found(det, frames):
    return [det.detect(f).ids.tolist() for f in frames]


@pytest.mark.parametrize(("dark", "floor"), [(0.6, 19), (0.45, 17)])
def test_shadow_edge(det, dark, floor):
    rng = np.random.default_rng(int(dark * 100))
    frames = [reading.render(rng, 10 * reading.CELLS, shadow=dark) for _ in range(FRAMES)]
    outer = sum(0 in ids for ids in _found(det, frames))
    print(f"shadow x{dark}: outer level in {outer}/{FRAMES}")
    assert outer >= floor


def test_occlusion_keeps_a_level(det):
    rng = np.random.default_rng(int(0.1 * 100) + 1)
    frames = [reading.render(rng, 10 * reading.CELLS, occlude=0.1) for _ in range(FRAMES)]
    anyl = sum(len(ids) > 0 for ids in _found(det, frames))
    print(f"occluded 10%: some level in {anyl}/{FRAMES}")
    assert anyl >= 9


def test_marker_like_negatives_stay_clean(det):
    rng = np.random.default_rng(7)
    hits = sum(len(det.detect(reading.negative(rng))) > 0 for _ in range(60))
    assert hits == 0
