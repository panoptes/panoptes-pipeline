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


def generate_network(num_units):
    """
    Generate simualted data from a network of PANOPTES units.
    :param num_units: the number of units to simulate
    :return:
    """
    # For every unit, let it observe a few stars, each on a random date, and output a PSC and light curve.
    stars_per_unit = random.randint(0, 5)
    for i in range(num_units):
        unit = "PAN{:03d}".format(i)
        site = random.choice(astroplan.get_site_names())
        for j in range(stars_per_unit):
            pic, coords = set_pic()
            start_time, end_time = set_obstime(coords, site)
            hdu = write_PSC(unit, pic, coords, start_time, end_time)
            write_lightcurve(hdu)


def set_pic():
    """
    Select a star to observe from the 2MASS catalog, and get its coordinates and name
    in the PANOPTES Input Catalog.
    :return: coordinates of PIC, PICID
    """
    # Eventually expand and read from file
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
    buffer = 1 #hours
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


def set_obstime(coords, site):
    """
    Based on the site of the unit, choose random times to begin and end observing the PIC
     within its rise and set.
    :param coords: coordinates of the PICm from SkyCoord
    :param site: name of the site the unit is at, from astroplan
    :return: randomized start and end time of observation
    """

    # Choose start and end for possible observing dates
    start = dt.strptime('2016-6-1', '%Y-%m-%d')
    end = dt.utcnow()
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


## Generate Fake Postage Stamp Cube (FITS cube)
def write_PSC(unit, pic, coords, start_time, end_time):
    """
    Generate a postage stamp cube with fake data for the given unit, PIC, and observation time.
    :param unit: ID of PANOPTES unit
    :param pic: PICID of star
    :param coords: coordinates of star, from SkyCoord
    :param start_time: start time of observation
    :param end_time: end time of observation
    :return: header data unit (hdu) of FITS file
    """
    # Set PSC info - fake field, camera, exposure time, and sequence ID.
    obstime = start_time
    camera = '0x2A'
    fields = ['x', 'y', 'z']
    target_name = 'field_{}'.format(random.choice(fields))
    seq_id = '{}{}_{}'.format(unit, camera, obstime.strftime('%Y%m%d_%H%M%SUT'))
    exptime = 100. # seconds
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
    data_cube = np.random.normal(sky_background, sky_sigma, (num_frames,ny,nx))
    hdu = fits.PrimaryHDU(data_cube)

    # Add metadata as FITS header and write to file
    hdu.header.extend(metadata)
    filename = "data/PSC_{}_{}".format(pic, seq_id)
    hdu.writeto(filename, clobber=True)

    return hdu


def write_lightcurve(hdu):
    """
    Generate a light curve from randomized flux values for each frame in the PSC.
    :param hdu: the FITS Header/Data Unit of the PSC
    :return:
    """

    pic = hdu.header["PICID"]
    stardir = 'data/{}'.format(pic)
    if not os.path.exists(stardir):
        os.makedirs(stardir)

    seq_id = hdu.header["SEQID"]
    filename = "{}/LC_{}_{}".format(stardir, pic, seq_id)
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
    parser = argparse.ArgumentParser(description='Generate a network of '
                                                 'simulated PANOPTES data.')
    parser.add_argument('num_units', type=int, nargs='?', default=1,
                        help='The number of PANOPTES units to simulate.')
    args = parser.parse_args()
    generate_network(args.num_units)
