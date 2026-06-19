"""Tests for nanofractal.generate — generate_aruco and generate_fractal.

TDD: this file is written BEFORE the implementation exists.
"""

from __future__ import annotations

import numpy as np
import pytest

import nanofractal as nf
import nanofractal._nanofractal as _nf

# ---------------------------------------------------------------------------
# generate_aruco
# ---------------------------------------------------------------------------


class TestGenerateArucoShape:
    @pytest.mark.parametrize("size_px", [100, 200, 500])
    def test_shape(self, size_px):
        img = nf.generate_aruco(0, size_px=size_px)
        assert img.shape == (size_px, size_px)

    def test_dtype(self):
        img = nf.generate_aruco(0, size_px=200)
        assert img.dtype == np.uint8

    def test_contiguous(self):
        img = nf.generate_aruco(0, size_px=200)
        assert img.flags["C_CONTIGUOUS"]


class TestGenerateArucoContent:
    def test_has_black_and_white(self):
        img = nf.generate_aruco(0, size_px=200)
        assert 0 in img
        assert 255 in img

    def test_different_marker_ids_differ(self):
        img0 = nf.generate_aruco(0, size_px=200)
        img1 = nf.generate_aruco(1, size_px=200)
        assert not np.array_equal(img0, img1)


class TestGenerateArucoBorderBits:
    def test_border_bits_2_has_wider_black_margin_than_1(self):
        # With border_bits=2 the outer black band should be wider.
        # Check that the corner regions are black and that border_bits=2
        # has more black pixels than border_bits=1.
        img1 = nf.generate_aruco(0, size_px=200, border_bits=1)
        img2 = nf.generate_aruco(0, size_px=200, border_bits=2)
        # Both have black corners
        assert img1[0, 0] == 0
        assert img2[0, 0] == 0
        # border_bits=2 has strictly more black pixels
        assert int(np.sum(img2 == 0)) > int(np.sum(img1 == 0))

    def test_default_border_bits_1_corner_is_black(self):
        img = nf.generate_aruco(0, size_px=200)
        assert img[0, 0] == 0
        assert img[0, -1] == 0
        assert img[-1, 0] == 0
        assert img[-1, -1] == 0


class TestGenerateArucoValidation:
    def test_size_px_zero_raises(self):
        with pytest.raises(ValueError, match="size_px"):
            nf.generate_aruco(0, size_px=0)

    def test_size_px_negative_raises(self):
        with pytest.raises(ValueError, match="size_px"):
            nf.generate_aruco(0, size_px=-1)

    def test_border_bits_zero_raises(self):
        with pytest.raises(ValueError, match="border_bits"):
            nf.generate_aruco(0, size_px=200, border_bits=0)

    def test_border_bits_negative_raises(self):
        with pytest.raises(ValueError, match="border_bits"):
            nf.generate_aruco(0, size_px=200, border_bits=-1)

    def test_invalid_marker_id_propagates(self):
        # Out-of-range id propagates as ValueError from the C++ helper
        with pytest.raises(ValueError):
            nf.generate_aruco(9999, size_px=200, dictionary=nf.Dict.DICT_4X4_50)

    def test_int_dictionary_accepted(self):
        # Plain int (not Dict enum) must also work
        img = nf.generate_aruco(0, size_px=100, dictionary=2)  # DICT_4X4_50 = 2
        assert img.shape == (100, 100)


# ---------------------------------------------------------------------------
# Round-trip: generate_aruco -> ArucoDetector
# ---------------------------------------------------------------------------


def _canvas(marker_img: np.ndarray, margin: int = 60) -> np.ndarray:
    """Paste marker on a white background with a quiet-zone margin."""
    s = marker_img.shape[0]
    c = np.full((s + 2 * margin, s + 2 * margin), 255, dtype=np.uint8)
    c[margin : margin + s, margin : margin + s] = marker_img
    return np.ascontiguousarray(c)


@pytest.mark.parametrize(
    "dictionary,marker_ids",
    [
        (nf.Dict.DICT_4X4_50, [0, 7, 42]),
        (nf.Dict.DICT_5X5_100, [5]),
    ],
)
def test_aruco_round_trip(dictionary, marker_ids):
    det = nf.ArucoDetector(dictionary, max_attempts=10)
    for mid in marker_ids:
        img = nf.generate_aruco(mid, size_px=300, dictionary=dictionary)
        canvas = _canvas(img, margin=80)
        res = det.detect(canvas)
        assert mid in res.ids.tolist(), (
            f"marker_id={mid} dict={dictionary.name} NOT detected; got ids={res.ids.tolist()}"
        )


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
