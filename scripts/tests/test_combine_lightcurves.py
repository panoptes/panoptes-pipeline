import pytest

from scripts.combine_lightcurves import LightCurveCombiner


@pytest.fixture
def combiner():
    """Return a valid light curve combiner"""
    return LightCurveCombiner(bucket='panoptes-simulated-data')


def test_storage_exists(combiner):
    """Tests that the combiner has an associated storage object"""
    assert combiner.storage is not None


def test_bad_pic(combiner):
    """Tests that error is raised when PIC id isn't recognized"""
    temp_dir = '/tmp/lc-combine'
    with pytest.raises(NameError):
        combiner.get_curves_for_pic('PIC_BAD', temp_dir)


def test_combine_curves(combiner):
    """Tests that simple 'curve' arrays are properly combined"""
    curves = [['a', 'b'], ['c', 'd']]
    master = combiner.combine_curves(curves)
    assert master == ['a', 'b', 'c', 'd']
