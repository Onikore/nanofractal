"""Tests for Task D: dict_grid_size and dict_num_markers public functions.

Values are verified against the vendored aruco_nano_v6.h (dictGridSize) and
the code tables in aruco_dicts.hpp / aruco_std_dicts.hpp.
"""

import pytest

import nanofractal as nf

# ---------------------------------------------------------------------------
# dict_num_markers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, expected",
    [
        (nf.Dict.DICT_4X4_50, 50),
        (nf.Dict.DICT_4X4_1000, 1000),
        (nf.Dict.ARUCO_MIP_36h12, 250),
        (nf.Dict.APRILTAG_36h11, 587),
    ],
)
def test_dict_num_markers(d, expected):
    assert nf.dict_num_markers(d) == expected


# ---------------------------------------------------------------------------
# dict_grid_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, expected",
    [
        (nf.Dict.DICT_4X4_50, 6),  # 4-bit inner + 2 border = 6
        (nf.Dict.DICT_5X5_50, 7),  # 5-bit inner + 2 border = 7
        (nf.Dict.DICT_6X6_50, 8),  # 6-bit inner + 2 border = 8
        (nf.Dict.DICT_7X7_50, 9),  # 7-bit inner + 2 border = 9
    ],
)
def test_dict_grid_size(d, expected):
    assert nf.dict_grid_size(d) == expected


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_dict_num_markers_invalid_raises():
    with pytest.raises(ValueError):
        nf.dict_num_markers(99)


def test_dict_grid_size_invalid_raises():
    with pytest.raises(ValueError):
        nf.dict_grid_size(99)


# ---------------------------------------------------------------------------
# Public surface: both names in __all__
# ---------------------------------------------------------------------------


def test_functions_in_all():
    assert "dict_grid_size" in nf.__all__
    assert "dict_num_markers" in nf.__all__
