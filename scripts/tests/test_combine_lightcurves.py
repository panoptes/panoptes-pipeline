import pytest

from scripts.combine_lightcurves import LightCurveCombiner
from scripts.tests.mock_storage import MockPanStorage


@pytest.fixture
def combiner():
    """Return a valid light curve combiner"""
    mock_storage = MockPanStorage(bucket_name='mock-bucket')
    return LightCurveCombiner(storage=mock_storage)


def test_storage_exists(combiner):
    """Tests that the combiner has an associated storage object"""
    assert combiner.storage is not None


def test_bad_pic(combiner):
    """Tests that error is raised when PIC id isn't recognized"""
    with pytest.raises(NameError):
        combiner.get_curves_for_pic('PIC_BAD')


def test_get_curves_from_pic(combiner):
    """Tests that curves from the mock data are properly returned"""
    curves = combiner.get_curves_for_pic('PIC_J0326137+295015')
    assert len(curves) > 0


def test_combine_curves(combiner):
    """Tests that simple 'curve' arrays are properly combined"""
    curves = [['a', 'b'], ['c', 'd']]
    master = combiner.combine_curves(curves)
    assert master == ['a', 'b', 'c', 'd']


def test_upload_output(combiner):
    """Tests that uploading basic json data works"""
    filename = 'test_file.json'
    data = ['a', 'b', 'c']
    combiner.upload_output(filename, data)
