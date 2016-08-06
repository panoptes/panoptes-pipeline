import os
import argparse
import numpy as np
import random
import json
import sys

import astroplan
from astroplan import Observer
from datetime import datetime
from datetime import timedelta
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError
from collections import defaultdict

from pocs.utils.google.storage import PanStorage


# Global variables
pan_storage = PanStorage(bucket='panoptes-simulated-data')
cameras = defaultdict(list)


def generate_network(num_units, start_date, end_date):
    """Generate simulated data from a network of PANOPTES units.

    :param num_units: the number of units to simulate
    :param start_date: the start date of observations
    :param end_date: the end date of observations
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    num_nights = (end_date - start_date).days
    if end_date < start_date:
        raise ValueError('End date must be after start date.')

    units = get_current_network()
    if len(units) == 0:
        print("Simulating new data network from {} units over {} nights. This make take "
              "a few minutes...".format(num_units, num_nights), file=sys.stdout)
        units = init_units(num_units)
    else:
        num_units = len(units)
        update_cameras()
        print('Adding new simulated data to current network of {} units over {} nights. This may '
              'take a few minutes...'.format(num_units, num_nights), file=sys.stdout)

    temp_dir = '/tmp/sim-data'

    # For every night, let every unit observe a few stars, and
    # output a Postage Stamp Cube (PSC) and light curve for each star.
    curr_date = start_date
    while curr_date <= end_date:
        for unit in units:
            site = random.choice(astroplan.get_site_names())
            stars_per_night = random.randint(0, 5)
            for j in range(stars_per_night):
                camera = random.choice(cameras[unit])
                field = get_field()
                pic, coords = get_pic()
                start_time, end_time = set_obs_time(coords, site, curr_date)

                psc_filename = "PSC/{}/{}/{}/{}.fits".format(
                    unit, camera, start_time.isoformat(), pic)
                lc_filename = "LC/{}/{}/{}/{}.json".format(pic, unit, camera,
                                                           start_time.isoformat())

                hdu = write_psc("{}/{}".format(temp_dir, psc_filename), unit, camera, field,
                                pic, coords, start_time, end_time)
                write_lightcurve("{}/{}".format(temp_dir, lc_filename), hdu)

                pan_storage.upload(
                    "{}/{}".format(temp_dir, psc_filename), remote_path=psc_filename)
                pan_storage.upload(
                    "{}/{}".format(temp_dir, lc_filename), remote_path=lc_filename)
            print('{} observed {} stars on {}.'.format(
                unit, stars_per_night, curr_date.strftime('%Y-%m-%d')))
        curr_date = curr_date + timedelta(days=1)


def init_units(num_units):
    """Initialize a network of new units and their cameras, assigning unique IDs."""
    units = []
    for i in range(num_units):
        unit = "PAN{:03d}".format(i)
        init_cameras(unit)
        units.append(unit)
    return units


def get_current_network():
    """Get the units and their cameras that currently have simulated data on the cloud."""
    units = []
    blobs = pan_storage.list_remote(prefix='LC')
    for blob in blobs:
        dirs = blob.name.split('/')
        for i in range(len(dirs)):
            dir = dirs[i]
            if dir.startswith('PAN'):
                unit = dir
                if unit not in units:
                    units.append(unit)
                cam = dirs[i + 1]
                if cam not in cameras:
                    cameras[unit].append(cam)
                break
    return units


def init_cameras(unit):
    """Set the camera IDs of the unit.

    :param unit: id of the unit
    """
    cams_per_unit = 2
    for i in range(cams_per_unit):
        add_new_camera(unit)


def add_new_camera(unit):
    """Add a new camera for the unit.

    Select a unique random camera id (first 6 digits of serial number.)
    If collisions occur too many times, raise an exception.
    :param: unit: the unit id to add a camera for
    """
    attempts = 0
    same = True
    while same:
        same = False
        cam = "{:06d}".format(random.randint(0, 999999))
        for un in cameras:
            for c in cameras[un]:
                if cam == c:
                    same = True
        if attempts > 100:
            raise Exception("Can't find unique camera ID.")
        attempts += 1
    cameras[unit].append(cam)
    return cam


def update_cameras():
    """Update the camera dictionary after getting current data network from cloud"""
    for unit in cameras:
        while len(cameras[unit]) < 2:
            add_new_camera(unit)


def get_field():
    """Return random (very fake) field name from list."""
    fields = ['field_x', 'field_y', 'field_z']
    return random.choice(fields)


def get_pic():
    """Return the PIC of a star to observe.

    Get a star from the 2MASS catalog, and its coordinates.
    Format its name in the PANOPTES Input Catalog (PIC).
    :return: coordinates of PIC, PICID
    """
    # Random stars I found as examples - eventually expand list and read from
    # file (?)
    star_list = ['2MASSW J0326137+295015',
                 '2MASSW J1632291+190441',
                 '2MASS J18365633+3847012']

    # Make sure PIC is valid and has coords found by astropy's SkyCoord.
    tries = 10
    for i in range(tries):
        star = random.choice(star_list)
        try:
            coords = SkyCoord.from_name(star, frame='fk5')
            pic = pic_name(star)
            return pic, coords
        except NameResolveError:
            print("Star name {} not found.".format(star))
    raise Exception("Too many tries to find a valid star.")


def random_time(rise_time, set_time):
    """Choose a random time to begin the observation between the object's rise and set.

    :param rise_time: time PIC rises
    :param set_time: time PIC sets
    :return: randomized start time of observation
    """
    buffer = 1  # hours, so that observing starts at least some amount of time before object set
    span = set_time - rise_time - timedelta(hours=buffer)
    return rise_time + timedelta(seconds=random.randint(0, span.seconds))


def random_duration(start_time, set_time):
    """Choose a random duration for the observation between the start and the time the object sets.

    :param start_time: time unit starts observing PIC
    :param set_time: time PIC sets
    :return: randomized duration of observation
    """
    min_dur = 1000
    span = set_time - start_time - timedelta(seconds=min_dur)
    return timedelta(seconds=random.randint(min_dur, min_dur + span.seconds))


def set_obs_time(coords, site, date):
    """Choose times to begin and end observing the PIC between its rise and set.

    :param coords: coordinates of the PIC object from SkyCoord
    :param site: name of the site the unit is at, from astroplan
    :param date: the date of the observation
    :return: randomized start and end time of observation
    """
    loc = Observer.at_site(site)

    # Use astroplan to get rise and set of star, and get reasonable random
    # obs_time and duration
    rise = loc.target_rise_time(date, coords, which='next')
    rise_time = datetime.strptime(rise.isot, "%Y-%m-%dT%H:%M:%S.%f")
    set = loc.target_set_time(rise_time, coords, which='next')
    set_time = datetime.strptime(set.isot, "%Y-%m-%dT%H:%M:%S.%f")
    start_time = random_time(rise_time, set_time)
    seq_duration = random_duration(start_time, set_time)
    end_time = start_time + seq_duration

    return start_time, end_time


def pic_name(star):
    """Format PICID for star from its catalog name."""
    jcoords = star.split()[1]
    pic = "PIC_{}".format(jcoords)
    return pic


def write_psc(filename, unit, camera, field, pic, coords, start_time, end_time):
    """Generate a postage stamp cube with fake data for the given unit, PIC, and observation time.

    :param filename: the path where the PSC will be stored
    :param unit: ID of PANOPTES unit
    :param camera: ID of camera used to take image sequence
    :param field: name of field of original image sequence
    :param pic: PICID of star
    :param coords: coordinates of star, from SkyCoord
    :param start_time: start time of observation
    :param end_time: end time of observation
    :return: Header/Data Unit (HDU) of FITS file
    """
    # Set PSC info - fake field, camera, exposure time, and sequence ID.
    obs_time = start_time
    target_name = 'field_{}'.format(field)
    seq_id = '{}_{}_{}'.format(
        unit, camera, obs_time.strftime('%Y%m%d_%H%M%SUT'))
    exptime = 100.  # seconds
    sky_background = 1000.
    sky_sigma = 5.
    nx = 12
    ny = 16

    # Choose random pixel position of postage stamp in original image
    xpixorg = random.randint(0, 5184 - nx)
    ypixorg = random.randint(0, 3456 - ny)

    # Set metadata
    metadata = {'SEQID': seq_id,
                'FIELD': target_name,
                'RA': coords.ra.to(u.degree).value,
                'DEC': coords.dec.to(u.degree).value,
                'EQUINOX': coords.equinox.value,
                'PICID': pic,
                'OBSTIME': obs_time.isoformat(),
                'XPIXORG': xpixorg,
                'YPIXORG': ypixorg,
                }

    # Set times of exposures and add to metadata, with slightly random time
    # gap between images
    frame = 0
    while obs_time < end_time:
        gap = timedelta(0, exptime + np.random.normal(5, 1))
        obs_time = obs_time + gap
        time = 'TIME{:04d}'.format(frame)
        exp_time = 'EXPT{:04d}'.format(frame)
        metadata[time] = obs_time.isoformat()
        metadata[exp_time] = exptime
        frame += 1
    num_frames = frame

    # Make random datacube from normal distribution, with same pixel dimensions as actual
    # (but no meaningful data)
    data_cube = np.random.normal(
        sky_background, sky_sigma, (num_frames, ny, nx))
    hdu = fits.PrimaryHDU(data_cube)

    # Add metadata as FITS header and write to file
    hdu.header.extend(metadata.items())
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    hdu.writeto(filename, clobber=True)

    return hdu


def write_lightcurve(filename, hdu):
    """Generate and write a JSON light curve from the PSC with randomized flux values.

    :param filename: the path where the light curve will be stored
    :param hdu: the FITS Header/Data Unit of the PSC
    :return: the light curve as a JSON object
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Make random relative flux and flux uncertainty data and write to JSON
    # file
    with open(filename, 'w') as FO:
        data = []
        for key in hdu.header:
            if key[:4] == "TIME":
                time = hdu.header[key]
                exptime = hdu.header['EXPT{}'.format(key[4:])]
                sig_r = 0.010
                sig_g = 0.006
                sig_b = 0.017
                r = np.random.normal(1, sig_r)
                g = np.random.normal(1, sig_g)
                b = np.random.normal(1, sig_b)
                entry = {
                    'time': time,
                    'exptime': exptime,
                    'R': r,
                    'G': g,
                    'B': b,
                    'sig_r': sig_r,
                    'sig_g': sig_g,
                    'sig_b': sig_b
                }
                data.append(entry)
        json.dump(data, FO)
    return json.dumps(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate data from a network of simulated PANOPTES units.')
    parser.add_argument('num_units', type=int, nargs='?', default=3,
                        help='The number of PANOPTES units to simulate.')
    parser.add_argument('start_date', type=str,
                        help='The start date of the simulated data, in Y-m-d format.')
    parser.add_argument('end_date', type=str,
                        help='The end date of the simulated data, in Y-m-d format.')
    parser.print_help()
    args = parser.parse_args()
    generate_network(args.num_units, args.start_date, args.end_date)
