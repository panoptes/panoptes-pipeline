import pytest
import re

from astropy import units as u
from datetime import datetime
from datetime import timedelta
from astropy.coordinates import SkyCoord

from scripts.generate_data_network import DataGenerator
from scripts.tests.mock_storage import MockPanStorage


@pytest.fixture()
def gen():
    mock_storage = MockPanStorage(bucket_name='mock-bucket')
    return DataGenerator(storage=mock_storage)


def test_generate_network_times_in_bounds(gen):
    """Tests end date is after start date"""
    t1 = '2016-1-1'
    t2 = '2016-1-5'
    with pytest.raises(ValueError):
        gen.generate_network(1, t2, t1)


def test_random_time_in_bounds(gen):
    """Tests that random observation time is between time bounds"""
    t1 = datetime(2016, 1, 1, 22, 0, 0)
    t2 = datetime(2016, 1, 2, 6, 0, 0)
    rand_time = gen.random_time(t1, t2)
    assert rand_time > t1
    assert rand_time < t2


def test_random_duration_in_bounds(gen):
    """Tests that the duration of observation is in bounds"""
    t1 = datetime(2016, 1, 1, 22, 0, 0)
    t2 = datetime(2016, 1, 2, 6, 0, 0)
    rand_dur = gen.random_duration(t1, t2)
    assert rand_dur < (t2 - t1)
    assert rand_dur > timedelta(seconds=1000)


def test_set_obs_time_in_bounds(gen):
    """Tests that the start and end time of observation are in bounds"""
    site = 'lick'
    coords = SkyCoord.from_name('2MASSW J0326137+295015', frame='fk5')
    date = datetime(2016, 1, 1)
    start_time, end_time = gen.set_obs_time(coords, site, date)
    assert start_time < end_time
    assert end_time - start_time > timedelta(seconds=1000)
    assert end_time - start_time < timedelta(hours=24)


def test_init_cameras(gen):
    """Tests that 2 cameras are created per unit"""
    units = ['PAN000', 'PAN001']
    for unit in units:
        gen.init_cameras(unit)
    for unit in gen.cameras:
        assert len(gen.cameras[unit]) == 2


def test_update_cameras(gen):
    """Tests that units have 2 cameras after update"""
    units = ['PAN000', 'PAN001']
    for unit in units:
        gen.init_cameras(unit)
    gen.update_cameras()
    for un in gen.cameras:
        assert len(gen.cameras[un]) == 2


def test_add_new_camera(gen):
    """Tests the new camera ID is in range, formatted properly and cam in dict"""
    unit = 'PAN000'
    cam = gen.add_new_camera(unit)
    assert int(cam) < 999999
    assert int(cam) > 0
    assert re.search('\d{6}', cam) is not None
    assert cam in gen.cameras[unit]


def test_get_pic_valid_coords(gen):
    """Tests that the star has valid coordinates given by SkyCoord"""
    pic, coords = gen.get_pic()
    assert coords.ra.to(u.degree).value >= 0
    assert coords.ra.to(u.degree).value <= 360
    assert abs(coords.dec.to(u.degree).value) <= 90


def test_pic_name(gen):
    """Tests that the PIC ID is formatted properly from the star name"""
    star = '2MASSW J0326137+295015'
    pic = gen.pic_name(star)
    assert pic.startswith('PIC')
    assert 'J0326137+295015' in pic


def test_get_field(gen):
    """Tests that field name is valid"""
    field = gen.get_field()
    assert field is not None


def test_build_psc(gen):
    """Test that the PSC written has all the correct fields with non-null values"""
    unit = 'PANXXX'
    camera = '000000'
    field = 'field_x'
    pic = 'PIC_J0326137+295015'
    coords = SkyCoord.from_name('2MASSW J0326137+295015', frame='fk5')
    start_time = datetime(2016, 1, 1, 22, 0, 0)
    end_time = datetime(2016, 1, 1, 23, 0, 0)
    hdu = gen.build_psc(unit, camera, field, pic, coords, start_time, end_time)
    keys = ['SEQID', 'FIELD', 'RA', 'DEC', 'EQUINOX', 'PICID',
            'OBSTIME', 'XPIXORG', 'YPIXORG', 'TIME0000', 'EXPT0000']
    assert hdu is not None
    for key in keys:
        assert key in hdu.header
        assert hdu.header[key] is not None


def test_build_lc(gen):
    """Test that a light curve is generated from a HDU with the correct fields"""
    unit = 'PANXXX'
    camera = '000000'
    field = 'field_x'
    pic = 'PIC_J0326137+295015'
    coords = SkyCoord.from_name('2MASSW J0326137+295015', frame='fk5')
    start_time = datetime(2016, 1, 1, 22, 0, 0)
    end_time = datetime(2016, 1, 2, 6, 0, 0)
    hdu = gen.build_psc(unit, camera, field, pic, coords, start_time, end_time)
    lc = gen.build_lightcurve(hdu)
    assert len(lc) > 0
    for data_point in lc:
        assert type(data_point['time']) is str
        assert type(data_point['exptime']) is float
        assert type(data_point['R']) is float
        assert type(data_point['G']) is float
        assert type(data_point['B']) is float
        assert type(data_point['sig_r']) is float
        assert type(data_point['sig_g']) is float
        assert type(data_point['sig_b']) is float


def test_get_current_network(gen):
    """Test that the currently stored network of units is properly retreived"""
    units = gen.get_current_network()
    assert len(units) == 1
    assert units[0] == 'PAN000'
