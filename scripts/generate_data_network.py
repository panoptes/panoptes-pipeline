import os
import argparse
import numpy as np
import random
import json

import astroplan
from astroplan import Observer
from datetime import datetime as dt
from datetime import timedelta as tdelta
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError
from collections import defaultdict


def generate_network(num_units, num_nights):
    """
    Generate simulated data from a network of PANOPTES units.
    :param num_units: the number of units to simulate
    :param num_nights: the number of nights the network has been observing
    :return:
    """
    # For every unit, set its info, let it observe a few stars,
    # each on a random date, and output a PSC and light curve.
    cameras = defaultdict(list)
    for i in range(num_units):
        unit = "PAN{:03d}".format(i)
        site = random.choice(astroplan.get_site_names())
        set_cameras(cameras, unit)

        stars_per_unit = random.randint(0, 5)
        for j in range(stars_per_unit):
            camera = random.choice(cameras[unit])
            field = get_field()
            pic, coords = set_pic()
            start_time, end_time = set_obstime(coords, site, num_nights)

            hdu = write_psc(unit, camera, field,
                            pic, coords, start_time, end_time)
            write_lightcurve(hdu)


def set_cameras(cam_dict, unitid):
    """
    Set the camera IDs of the unit
    :param cam_dict: dictionary mapping unitid to camera IDs
    :param unitid: id of the unit
    :return:
    """
    # Try to create a unique camera id. If collisions occur too
    # many times, raise an exception.
    cams_per_unit = 2
    for i in range(cams_per_unit):
        attempts = 0
        same = True
        while same:
            same = False
            cam = "{:06d}".format(random.randint(0, 999999))
            for un in cam_dict:
                for c in cam_dict[un]:
                    if cam == c:
                        same = True
            if attempts > 100:
                raise Exception("Can't find unique camera ID.")
            attempts += 1
        cam_dict[unitid].append(cam)


def get_field():
    """
    Get random (very fake) field name from list
    :return: field name
    """
    fields = ['field_x', 'field_y', 'field_z']
    return random.choice(fields)


def set_pic():
    """
    Select a star to observe from the 2MASS catalog, and get its coordinates and name
    in the PANOPTES Input Catalog.
    :return: coordinates of PIC, PICID
    """
    # Random stars I found as examples - eventually expand list and read from file (?)
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


def random_date(start, end):
    """
    Get a random date to observe on.
    :param start: first possible observation date
    :param end: last possible observation date
    :return: randomized date between start and end
    """
    span = end - start
    return start + tdelta(days=random.randint(0, span.days))


def random_time(rise_time, set_time):
    """
    Choose a random time to begin the observation between the object's rise and set.
    :param rise_time: time PIC rises
    :param set_time: time PIC sets
    :return: randomized start time of observation
    """
    buffer = 1  # hours, so that observing starts at least some amount of time before object set
    span = set_time - rise_time - tdelta(hours=buffer)
    return rise_time + tdelta(seconds=random.randint(0, span.seconds))


def random_duration(start_time, set_time):
    """
    Choose a random duration for the observation between the start and the time the object sets.
    :param start_time: time unit starts observing PIC
    :param set_time: time PIC sets
    :return: randomized duration of observation
    """
    min_dur = 1000
    span = set_time - start_time - tdelta(seconds=min_dur)
    return tdelta(seconds=random.randint(min_dur, min_dur+span.seconds))


def set_obstime(coords, site, nights):
    """
    Based on the site of the unit, choose random times to begin and end observing the PIC
     within its rise and set.
    :param coords: coordinates of the PICm from SkyCoord
    :param site: name of the site the unit is at, from astroplan
    :return: randomized start and end time of observation
    """

    # Choose start and end for possible observing dates
    end = dt.utcnow()
    start = end - tdelta(days=nights)
    date = random_date(start, end)
    loc = Observer.at_site(site)

    # Use astroplan to get rise and set of star, and get reasonable random obstime and duration
    rise = loc.target_rise_time(date, coords, which='next')
    rise_time = dt.strptime(rise.isot, "%Y-%m-%dT%H:%M:%S.%f")
    set = loc.target_set_time(rise_time, coords, which='next')
    set_time = dt.strptime(set.isot, "%Y-%m-%dT%H:%M:%S.%f")
    start_time = random_time(rise_time, set_time)
    seq_duration = random_duration(start_time, set_time)
    end_time = start_time + seq_duration

    return start_time, end_time


def pic_name(star):
    """
    Write PICID for star from its catalog name
    :param star: ID of star in catalog
    :return: PICID based on its coordinates as in the catalog
    """
    jcoords = star.split()[1]
    pic = "PIC_{}".format(jcoords)
    return pic


def write_psc(unit, camera, field, pic, coords, start_time, end_time):
    """
    Generate a postage stamp cube with fake data for the given unit, PIC, and observation time.
    :param unit: ID of PANOPTES unit
    :param camera: ID of camera used to take image sequence
    :param field: name of field of original image sequence
    :param pic: PICID of star
    :param coords: coordinates of star, from SkyCoord
    :param start_time: start time of observation
    :param end_time: end time of observation
    :return: header data unit (hdu) of FITS file
    """
    # Set PSC info - fake field, camera, exposure time, and sequence ID.
    obstime = start_time
    target_name = 'field_{}'.format(field)
    seq_id = '{}_{}_{}'.format(unit, camera, obstime.strftime('%Y%m%d_%H%M%SUT'))
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
                'OBSTIME': obstime.isoformat(),
                'XPIXORG': xpixorg,
                'YPIXORG': ypixorg,
                }

    # Set times of exposures and add to metadata, with slightly random time gap between images
    t = 0
    while obstime < end_time:
        gap = tdelta(0, exptime + np.random.normal(5,1))
        obstime = obstime + gap
        metadata['TIME{:04d}'.format(t)] = obstime.isoformat()
        metadata['EXPT{:04d}'.format(t)] = exptime
        t += 1
    num_frames = t

    # Make random datacube from normal distribution, with same pixel dimensions as actual (but no meaningful data)
    data_cube = np.random.normal(sky_background, sky_sigma, (num_frames, ny, nx))
    hdu = fits.PrimaryHDU(data_cube)

    # Add metadata as FITS header and write to file
    hdu.header.extend(metadata)
    filename = "data/PSC/{}/{}/{}/{}.fits".format(unit, camera, start_time.isoformat(), pic)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    hdu.writeto(filename, clobber=True)

    return hdu


def write_lightcurve(hdu):
    """
    Generate a light curve from randomized flux values for each frame in the PSC
    and write it to a JSON file.
    :param hdu: the FITS Header/Data Unit of the PSC
    :return:
    """

    # Get info for filename
    pic = hdu.header["PICID"]
    seqid = hdu.header["SEQID"]
    unit = seqid.split('_')[0]
    camera = seqid.split('_')[1]
    start_time = hdu.header["OBSTIME"]
    filename = "data/LC/{}/{}/{}/{}.json".format(pic, unit, camera, start_time)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Make random relative flux and flux uncertainty data and write to JSON file
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data from a network of '
                                                 'simulated PANOPTES units.')
    parser.add_argument('num_units', type=int, nargs='?', default=5,
                        help='The number of PANOPTES units to simulate.')
    parser.add_argument('num_nights', type=int, nargs='?', default=5,
                        help='The number of nights to simulate, starting continuously before today.')
    args = parser.parse_args()
    generate_network(args.num_units, args.num_nights)
