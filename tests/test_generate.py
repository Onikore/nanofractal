"""Tests for nanofractal.generate — generate_fractal."""

from __future__ import annotations

import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _canvas(marker_img: np.ndarray, margin: int = 60) -> np.ndarray:
    """Paste marker on a white background with a quiet-zone margin."""
    s = marker_img.shape[0]
    c = np.full((s + 2 * margin, s + 2 * margin), 255, dtype=np.uint8)
    c[margin : margin + s, margin : margin + s] = marker_img
    return np.ascontiguousarray(c)


# ---------------------------------------------------------------------------
# generate_fractal
# ---------------------------------------------------------------------------


class TestGenerateFractalShape:
    def test_shape(self):
        img = nf.generate_fractal("FRACTAL_5L_6", size_px=400)
        assert img.shape == (400, 400)

    def test_dtype(self):
        img = nf.generate_fractal("FRACTAL_5L_6", size_px=400)
        assert img.dtype == np.uint8

    def test_contiguous(self):
        img = nf.generate_fractal("FRACTAL_5L_6")
        assert img.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize("size_px", [200, 400, 600])
    def test_parametric_sizes(self, size_px):
        img = nf.generate_fractal("FRACTAL_5L_6", size_px=size_px)
        assert img.shape == (size_px, size_px)


class TestGenerateFractalContent:
    def test_has_black_and_white(self):
        img = nf.generate_fractal("FRACTAL_5L_6", size_px=400)
        assert 0 in img
        assert 255 in img


class TestGenerateFractalValidation:
    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            nf.generate_fractal("NOT_A_CONFIG")

    def test_size_px_zero_raises(self):
        with pytest.raises(ValueError):
            nf.generate_fractal("FRACTAL_5L_6", size_px=0)

    def test_size_px_negative_raises(self):
        with pytest.raises(ValueError):
            nf.generate_fractal("FRACTAL_5L_6", size_px=-1)

    @pytest.mark.parametrize(
        "config", ["FRACTAL_2L_6", "FRACTAL_3L_6", "FRACTAL_4L_6", "FRACTAL_5L_6"]
    )
    def test_all_valid_configs_accepted(self, config):
        img = nf.generate_fractal(config, size_px=200)
        assert img.shape == (200, 200)


# ---------------------------------------------------------------------------
# Round-trip: generate_fractal -> FractalDetector
# ---------------------------------------------------------------------------


def test_fractal_round_trip():
    config = "FRACTAL_5L_6"
    ext_id = _nf._fractal_external_id(config)
    img = nf.generate_fractal(config, size_px=520)
    canvas = _canvas(img, margin=100)
    det = nf.FractalDetector(config)
    res = det.detect(canvas)
    assert ext_id in res.ids.tolist(), (
        f"external id={ext_id} NOT detected; got ids={res.ids.tolist()}"
    )


# ---------------------------------------------------------------------------
# Nested composite: every level must be present and decodable
# ---------------------------------------------------------------------------

_ALL_LEVELS = {
    "FRACTAL_2L_6": [0, 1],
    "FRACTAL_3L_6": [0, 1, 2],
    "FRACTAL_4L_6": [0, 1, 2, 3],
    "FRACTAL_5L_6": [0, 1, 2, 3, 4],
}


@pytest.mark.parametrize("config", sorted(_ALL_LEVELS))
def test_nested_round_trip_detects_every_level(config):
    # Large enough that the innermost level has ~40 px sides.
    size = 2400 if config == "FRACTAL_5L_6" else 1200
    canvas = _canvas(nf.generate_fractal(config, size_px=size), margin=100)
    res = nf.FractalDetector(config).detect(canvas)
    assert sorted(res.ids.tolist()) == _ALL_LEVELS[config]


def test_nested_false_is_external_level_only():
    ext = nf.generate_fractal("FRACTAL_5L_6", size_px=520, nested=False)
    full = nf.generate_fractal("FRACTAL_5L_6", size_px=520)
    assert ext.shape == full.shape
    assert not np.array_equal(ext, full)  # inner levels add detail
    # outside the external level's hole the two renders agree
    np.testing.assert_array_equal(ext[:120], full[:120])


def test_nested_binary_values_only():
    img = nf.generate_fractal("FRACTAL_4L_6", size_px=333)
    assert set(np.unique(img).tolist()) <= {0, 255}
