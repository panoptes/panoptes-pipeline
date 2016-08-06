import pytest
import re

from collections import defaultdict
from astropy import units as u
from datetime import datetime
from datetime import timedelta

from scripts import generate_data_network as gen


def test_generate_network_times_in_bounds():
    """Tests end date is after start date"""
    t1 = '2016-1-1'
    t2 = '2016-1-5'
    with pytest.raises(ValueError):
        gen.generate_network(1, t2, t1)


def test_random_time_in_bounds():
    """Tests that random observation time is between time bounds"""
    t1 = datetime(2016, 1, 1, 22, 0, 0)
    t2 = datetime(2016, 1, 2, 6, 0, 0)
    rand_time = gen.random_time(t1, t2)
    assert rand_time > t1
    assert rand_time < t2


def test_random_duration_in_bounds():
    """Tests that the duration of observation is in bounds"""
    t1 = datetime(2016, 1, 1, 22, 0, 0)
    t2 = datetime(2016, 1, 2, 6, 0, 0)
    rand_dur = gen.random_duration(t1, t2)
    assert rand_dur < (t2 - t1)
    assert rand_dur > timedelta(seconds=1000)


def test_init_cameras():
    """Tests that 2 cameras are created per unit"""
    cameras = defaultdict(list)
    units = ['PAN000', 'PAN001']
    for unit in units:
        gen.init_cameras(unit)
    for unit in cameras:
        assert len(cameras[unit]) == 2


def test_update_cameras():
    """Tests that units have 2 cameras after update"""
    cameras = defaultdict(list)
    units = ['PAN000', 'PAN001']
    for unit in units:
        gen.init_cameras(unit)
    gen.update_cameras()
    for un in cameras:
        assert len(cameras[un]) == 2


def test_add_new_camera():
    """Tests the new camera ID is in range and formatted properly"""
    unit = 'PAN000'
    cam = gen.add_new_camera(unit)
    assert int(cam) < 999999
    assert int(cam) > 0
    assert re.search('\d{6}', cam) is not None


def test_get_pic_valid_coords():
    """Tests that the star has valid coordinates given by SkyCoord"""
    pic, coords = gen.get_pic()
    assert coords.ra.to(u.degree).value >= 0
    assert coords.ra.to(u.degree).value <= 360
    assert abs(coords.dec.to(u.degree).value) <= 90


def test_pic_name():
    """Tests that the PIC ID is formatted properly from the star name"""
    star = '2MASSW J0326137+295015'
    pic = gen.pic_name(star)
    assert pic.startswith('PIC')
    assert 'J0326137+295015' in pic


def test_get_field():
    """Tests that field name is valid"""
    field = gen.get_field()
    assert field is not None
