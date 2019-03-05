import pytest

from piaa.utils import helpers


def test_pixel_color():
    assert helpers.pixel_color(0, 0) == 'G2'
    assert helpers.pixel_color(0, 1) == 'R'
    assert helpers.pixel_color(1, 0) == 'B'
    assert helpers.pixel_color(1, 1) == 'G1'
