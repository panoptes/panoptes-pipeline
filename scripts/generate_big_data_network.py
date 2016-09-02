import os
import argparse
import numpy as np
import random
import json
import sys

import astroplan
from astroplan import Observer
from astroplan import download_IERS_A
from datetime import datetime
from datetime import timedelta
from astropy.time import Time
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.angles import Angle
from astropy.coordinates import EarthLocation
from collections import defaultdict

from pocs.utils.google.storage import PanStorage


#############################################################################
# This class is a simulator that is designed to simulate larger quantities  #
# of data, more accurate in the levels of data products than                #
# generate_data_network.py (which has more realistic star data). Use this   #
# to simulate large quantities of data to test the pipeline. Should         #
# eventually be merged with generate_data_network.py.                       #
#############################################################################

class DataGenerator(object):
    """Class to generate a network of simulated PANOPTES images and light curves

    Now just a wrapper class, should be updated to abstract out other objects eventually
    """

    def __init__(self, storage=None, local_dir=None):
        if not storage:
            storage = PanStorage(bucket_name='panoptes-simulated-data')
        if not local_dir:
            local_dir = '/tmp/sim-data'
        self.storage = storage
        self.cameras = defaultdict(list)
        self.star_dict = {}
        self.unit_dict = {}
        self.local_dir = local_dir
        download_IERS_A()

    def generate_network(self, num_units, start_date, end_date, cloud):
        """Generate simulated data from a network of PANOPTES units.

        :param num_units: the number of units to simulate
        :param start_date: the start date of observations
        :param end_date: the end date of observations
        """
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        num_nights = (end_date - start_date).days + 1
        if end_date < start_date:
            raise ValueError('End date must be after start date.')

        units = self.get_current_network()
        if len(units) == 0:
            print("Simulating new data network from {} units over {} nights. This make take "
                  "a few minutes...".format(num_units, num_nights), file=sys.stdout)
            units = self.init_units(num_units)
        else:
            num_units = len(units)
            self.update_cameras()
            print('Adding new simulated data to current network of {} units over {} nights. This may '
                  'take a few minutes...'.format(num_units, num_nights), file=sys.stdout)

        # For every night, let every unit observe a few stars, and
        # output a Postage Stamp Cube (PSC) and light curve for each star.
        lc_count = 0
        curr_date = start_date
        while curr_date <= end_date:
            for unit in units:

                # Start first sequence 1-2 hours after sunset
                site = self.unit_dict[unit]
                loc = Observer.at_site(site)
                sunset = loc.sun_set_time(
                    Time(curr_date, scale='utc'), which='next')
                sunset_time = datetime.strptime(
                    sunset.isot, "%Y-%m-%dT%H:%M:%S.%f")
                last_obs_end_time = sunset_time + \
                    timedelta(minutes=(random.random() * 60 + 60))
                sequences_per_night = random.randint(4, 6)
                for i in range(sequences_per_night):

                    stars_per_sequence = random.randint(100, 200)
                    start_time, end_time = self.set_obs_time(last_obs_end_time)
                    last_obs_end_time = end_time

                    for j in range(stars_per_sequence):

                        # Get observation information
                        field = self.get_field()
                        pic, coords = self.get_fake_pic()

                        for camera in self.cameras[unit]:
                            psc_filename = "PSC/{}/{}/{}/{}.fits".format(
                                unit, camera, start_time.isoformat(), pic)
                            lc_filename = "LC/{}/{}/{}/{}.json".format(pic, unit, camera,
                                                                       start_time.isoformat())

                            # Create data products
                            hdu = self.build_psc(unit, camera, field,
                                                 pic, coords, start_time, end_time)
                            lc = self.build_lightcurve(hdu)

                            # Write data products to local temp files
                            self.write_psc(
                                "{}/{}".format(self.local_dir, psc_filename), hdu)
                            self.write_lightcurve(
                                "{}/{}".format(self.local_dir, lc_filename), lc)

                            # Upload data products from local files to cloud
                            if cloud:
                                self.storage.upload(
                                    "{}/{}".format(self.local_dir, psc_filename), remote_path=psc_filename)
                                self.storage.upload(
                                    "{}/{}".format(self.local_dir, lc_filename), remote_path=lc_filename)
                                lc_count += 1

                            print('{}|LC{}| {}/{} observed star {} on {}.'.format(
                                    datetime.now().time(), lc_count, unit, camera, pic, curr_date.strftime('%Y-%m-%d')))

                stars_per_night = sequences_per_night * stars_per_sequence
                print('{} {} observed {} stars on {}.'.format(
                    datetime.now().time(), unit, stars_per_night, curr_date.strftime('%Y-%m-%d')))
            curr_date = curr_date + timedelta(days=1)

    def init_units(self, num_units):
        """Initialize a network of new units and their cameras, assigning unique IDs."""
        units = []
        for i in range(num_units):
            unit = "PAN{:03d}".format(i)
            site = random.choice(EarthLocation.get_site_names())
            self.unit_dict[unit] = site
            self.init_cameras(unit)
            units.append(unit)
        return units

    def get_current_network(self):
        """Get the units and their cameras that currently have simulated data on the cloud."""
        units = []
        files = self.storage.list_remote(prefix='LC')
        for fl in files:
            dirs = fl.split('/')
            for i in range(len(dirs)):
                dir = dirs[i]
                if dir.startswith('PAN'):
                    unit = dir
                    if unit not in units:
                        units.append(unit)
                        site = random.choice(EarthLocation.get_site_names())
                        self.unit_dict[unit] = site
                    cam = dirs[i + 1]
                    if cam not in self.cameras[unit]:
                        self.cameras[unit].append(cam)
                    break
        return units

    def init_cameras(self, unit):
        """Set the camera IDs of the unit.

        :param unit: id of the unit
        """
        cams_per_unit = 2
        for i in range(cams_per_unit):
            self.add_new_camera(unit)

    def add_new_camera(self, unit):
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
            for un in self.cameras:
                for c in self.cameras[un]:
                    if cam == c:
                        same = True
            if attempts > 100:
                raise Exception("Can't find unique camera ID.")
            attempts += 1
        self.cameras[unit].append(cam)
        return cam

    def update_cameras(self):
        """Update the camera dictionary after getting current data network from cloud"""
        for unit in self.cameras:
            while len(self.cameras[unit]) < 2:
                self.add_new_camera(unit)

    def get_field(self):
        """Return random (very fake) field name from list."""
        fields = ['field_x', 'field_y', 'field_z']
        return random.choice(fields)

    def get_fake_pic(self):
        """Generate a fake Panoptes Input Catalog star and random coordinates."""
        pic = "PIC_{:06d}".format(random.randint(0, 50))
        if pic in self.star_dict:
            coords = self.star_dict[pic]
        else:
            ra = Angle((random.random() * 360) * u.deg)
            dec = Angle((random.random() * 180 - 90) * u.deg)
            coords = SkyCoord(ra.to(u.degree), dec.to(u.degree), frame='fk5')
            self.star_dict[pic] = coords
        return pic, coords

    def set_obs_time(self, last_obs_end_time):
        """Choose times to begin and end observing the PIC between its rise and set.

        :param coords: coordinates of the PIC object from SkyCoord
        :param site: name of the site the unit is at, from astroplan
        :param date: the date of the observation
        :return: randomized start and end time of observation
        """
        # Set 0-10 minute buffer between end of last observation and start of
        # new observation
        buffer_new_obs = timedelta(minutes=random.random() * 10)
        start_time = last_obs_end_time + buffer_new_obs
        # Random sequence duration between 1 and 3 hours
        seq_duration = timedelta(minutes=(random.random() * 120) + 60)
        end_time = start_time + seq_duration
        return start_time, end_time

    def build_psc(self, unit, camera, field, pic, coords, start_time, end_time):
        """Generate a postage stamp cube with fake data for the given unit, PIC, and observation time.

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

        return hdu

    def write_psc(self, filename, hdu):
        """Write PSC to file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        hdu.writeto(filename, clobber=True)

    def build_lightcurve(self, hdu):
        """Generate and write a JSON light curve from the PSC with randomized flux values.

        :param hdu: the FITS Header/Data Unit of the PSC
        :return: the light curve as a JSON object
        """
        # Make random relative flux and flux uncertainty data
        lc = []
        for key in hdu.header:
            if key[:4] == "TIME":
                time = hdu.header[key]
                exptime = hdu.header['EXPT{}'.format(key[4:])]
                seq_id = hdu.header['SEQID']
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
                    'sig_b': sig_b,
                    'seq_id': seq_id
                }
                lc.append(entry)
        return lc

    def write_lightcurve(self, filename, lc):
        """Write light curve to file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as FO:
            json.dump(lc, FO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate data from a network of simulated PANOPTES units.')
    parser.add_argument('num_units', type=int, nargs='?', default=3,
                        help='The number of PANOPTES units to simulate.')
    parser.add_argument('start_date', type=str,
                        help='The start date of the simulated data, in Y-m-d format.')
    parser.add_argument('end_date', type=str,
                        help='The end date of the simulated data, in Y-m-d format.')
    parser.add_argument('local_dir', type=str, help='The local directory in which to store the simulated data.')
    parser.add_argument('-c', '--cloud', action='store_true', help='Upload simulated data to Google Cloud Storage.')
    parser.print_help()
    args = parser.parse_args()
    gen = DataGenerator(local_dir=args.local_dir)
    gen.generate_network(args.num_units, args.start_date, args.end_date, args.cloud)
