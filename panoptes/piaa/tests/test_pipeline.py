import pytest

from panoptes.piaa import pipeline


def test_get_imag():
    assert pipeline.get_imag(10) == -2.5
