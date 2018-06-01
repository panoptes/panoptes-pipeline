import os
import shutil
import subprocess

from warnings import warn
from collections import namedtuple
from glob import glob

import h5py
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time

from scipy.sparse.linalg import lsmr

from dateutil.parser import parse as date_parse

from tqdm import tqdm

from piaa.utils import make_masks
from piaa.utils import helpers
from piaa import exoplanets

from pocs.utils.images import fits as fits_utils
from pocs.utils.error import SolveError

import logging

DATA_DIR = '/var/panoptes/images/fields'

PSC = namedtuple('PSC', ['data', 'mask'])


class Observation(object):

    def __init__(self,
                 sequence,
                 data_dir=DATA_DIR,
                 *args,
                 **kwargs):
        """ A sequence of images to be processed as one observation """

        self._data_dir = os.path.join(DATA_DIR, sequence)
        os.makedirs(self._data_dir, exist_ok=True)

        log_file = os.path.join(self._data_dir, 'processing.log')
        logging.basicConfig(filename=log_file, level=logging.DEBUG,
                            format='%(asctime)s %(message)s')

        try:
            os.remove('/var/panoptes/logs/processing.log')
        except FileNotFoundError:
            pass
        finally:
            os.symlink(log_file, '/var/panoptes/logs/processing.log')

        logging.info('*' * 80)
        logging.info('Setting up Observation for analysis - {}'.format(sequence))

        self.sequence = sequence
        self.unit_id, self.field, self.cam_id, self.seq_time = self.sequence.split('/')

        self._wcs = None
        self._planet = None
        self._transit_info = None

        super(Observation, self).__init__()

        self._img_h = 3476
        self._img_w = 5208

        self.stamp_size = (None, None)

        self._point_sources = None

        self._image_times = None
        self._total_integration_time = None

        self._rgb_masks = None

        self._hdf5_stamps = None
        self._hdf5_stamps_fn = self.data_dir + '.hdf5'

        self.target_slice = None

        self._num_frames = 0

    @property
    def planet(self):
        if self._planet is None:
            # Holds some properties about the planet
            self._planet = exoplanets.Exoplanet(self.field)

        return self._planet

    @property
    def hdf5_stamps(self):
        if self._hdf5_stamps is None:
            self._hdf5_stamps = h5py.File(self._hdf5_stamps_fn, 'a')

        return self._hdf5_stamps

    @property
    def stamps(self):

        try:
            stamp_group = self.hdf5_stamps['stamps']
        except KeyError:
            logging.debug("Creating stamps file.")
            stamp_group = self.hdf5_stamps.create_group('stamps')

        return stamp_group

    @property
    def point_sources(self):
        if self._point_sources is None:
            self.lookup_point_sources()

        return self._point_sources

    @property
    def data_dir(self):
        """ Image directory containing FITS files

        When setting a new image directory, FITS files are automatically
        loaded into the `files` property

        Returns:
            str: Path to image directory
        """
        return self._data_dir

    @property
    def image_times(self):
        if self._image_times is None:
            image_times = dict()

            for fn in self.files:
                # Get information about each image
                try:
                    _, _, _, _, _, unit_id, field, cam_id, seq_id, img_id = fn.split('/')
                except ValueError as e:
                    logging.warning(e)
                    continue

                # Get the time from the image
                img_id = img_id.split('.')[0]
                try:
                    t0 = Time(date_parse(img_id), format='datetime')
                except Exception as e:
                    logging.warning(e)
                    continue

                p = self.planet.get_phase(t0)

                image_times[img_id] = p

            self._image_times = image_times

        return self._image_times

    @property
    def total_integration_time(self):
        if self._total_integration_time is None:
            self._total_integration_time = np.array(
                [np.round(float(fits.getval(f, 'EXPTIME'))) for f in self.files]).sum()

        return self._total_integration_time

    @property
    def wcs(self):
        if self._wcs is None:
            try:
                self._wcs = WCS(self.files[0])
            except IndexError:
                self._wcs = WCS(self.files[0], axis=1)

        return self._wcs

    @property
    def num_frames(self):
        return self._num_frames

    @num_frames.setter
    def num_frames(self, num):
        self._num_frames = num

    @property
    def num_point_sources(self):
        return len(self.point_sources)

    @property
    def rgb_masks(self):
        """ RGB Mask arrays

        Read the RGB masks from a stored file or generate and save to file accordingly

        Returns:
            numpy.MaskedArray: A 3xNxM array where the first axis corresponds to color
                and the NxM is the full frame size
        """
        if self._rgb_masks is None:
            rgb_mask_file = '{}/rgb_masks.npz'.format(os.getenv('PANDIR'))
            try:
                self._rgb_masks = np.load(rgb_mask_file)
            except FileNotFoundError:
                logging.debug("Making RGB masks")
                self._rgb_masks = np.array(make_masks(self.data_cube[0]))
                self._rgb_masks.dump(rgb_mask_file)

        return self._rgb_masks

    @property
    def transit_info(self):
        if self._transit_info is None:
            for fn in self.files:
                # Get information about each image
                try:
                    _, _, _, _, _, unit_id, field, cam_id, seq_id, img_id = fn.split('/')
                except ValueError as e:
                    logging.warning(e)
                    continue

                # Get the time from the image
                img_id = img_id.split('.')[0]
                try:
                    t0 = Time(date_parse(img_id), format='datetime')
                except Exception as e:
                    logging.warning(e)
                    continue

                # Determine if in transit
                try:
                    in_t = self.planet.in_transit(t0, with_times=True)
                except Exception as e:
                    logging.warning(e)
                    continue

                if in_t[0]:
                    self._transit_info = in_t[1]
                    break

        return self._transit_info

    def get_wcs(self, frame):
        hdu_axis = 0

        fn = self.files[frame]
        if fn.endswith('.fz'):
            hdu_axis = 1

        return WCS(fn, axis=hdu_axis)

    def get_header_value(self, frame_index, header):
        return fits.getval(self.files[frame_index], header)

    def get_psc(self, source, frame_slice=None, remove_bias=False):
        try:
            psc = np.array(self.stamps[source])
        except KeyError:
            raise Exception("You must run create_stamps first")

        if frame_slice is not None:
            psc = psc[frame_slice]

        if remove_bias:
            psc -= self.camera_bias

        return psc

    def get_stamps(
            self,
            stamp_size=(10, 10),
            cleanup_after=True,
            remove_stamps_file=False,
            upload=True,
            force_new=False,
            *args, **kwargs):
        """Makes or gets PANOPTES Stamps Cubes (PSC) file.

        This will first look for the HDF5 stamps file locally, then in the
        storage bucket. If not found, will download all the relevant FITS
        files, make sure they are plate-solved, then upload the stamps file
        back to bucket when done.

        Note:
            This can be a long running process!

        Args:
            stamp_size (tuple, optional): Size of stamps, default (10, 10). The
                size is for individual pixels. Stamps should have an odd number of
                superpixels, meaning an even number of individuals pixels that in
                integer increments of four (4), e.g. (6, 6), (10, 10), (14, 14).
            cleanup_after (bool, optional): If files should be removed afterward,
                default True.
            remove_stamps_file (bool, optional): If the generated stamp file should
                also be removed during cleanup, default False.
            upload (bool, optional): Upload stamps to storage bucket, default True.
            force_new (bool, optional): If a new stamps file should be created,
                default False.
        """
        logging.info('Creating stamps')

        # Check for file locally
        if os.path.exists(self._hdf5_stamps_fn):
            logging.debug('Using local stamps file')
            return

        # Check if in storage bucket
        stamp_blob = helpers.get_observation_blobs(self.sequence + '.hdf5')
        if stamp_blob:
            logging.info('Downloading stamps file from storage bucket')
            helpers.download_blob(stamp_blob, save_as=self._hdf5_stamps_fn)
            return

        # Download FITS files
        logging.debug('Downloading FITS files')
        fits_blobs = helpers.get_observation_blobs(self.sequence)

        # Download all the FITS files from a bucket
        self.files = list()
        if fits_blobs:
            for blob in tqdm(fits_blobs, desc='Downloading FITS files'.ljust(25)):
                fits_fn = helpers.unpack_blob(blob, save_dir=self._data_dir)
                self.files.append(fits_fn)

        self.num_frames = len(self.files)

        # Plate-solve all the images - safe to run again
        logging.debug('Plate-solving FITS files')
        for fn in tqdm(self.files, desc='Solving files'.ljust(25)):
            try:
                fits_utils.get_solve_field(fn, timeout=90)
            except SolveError:
                logging.warning("Can't solve file {}".format(fn))
                logging.debug("Stopping processing for sequence and cleaning up")
                self._do_cleanup(remove_stamps_file=True)

        # Lookup point sources
        # You need to set the env variable for the password for TESS catalog DB (ask Wilfred)
        # os.environ['PGPASSWORD'] = 'sup3rs3cr3t'
        logging.debug('Looking up point sources via TESS catalog')
        self.lookup_point_sources(use_sextractor=False, use_tess_catalog=True)
        logging.debug("Number of sources detected: {}".format(len(self.point_sources)))

        # Create stamps
        logging.debug('Creating stamps')
        self.create_stamp_slices(stamp_size=stamp_size)

        # Upload to storage bucket
        if upload:
            logging.debug('Uploading stamps file to storage bucket')
            helpers.upload_to_bucket(self._hdf5_stamps_fn, self.sequence + '.hdf5')

        # Cleanup
        if cleanup_after:
            self._do_cleanup(**kwargs)

    def _do_cleanup(self, remove_stamps_file=False):
        logging.debug('Cleaning up FITS files')
        for fn in self.files:
            os.remove(fn)
            try:
                os.remove(fn.replace('.fits', '.solved'))
            except Exception:
                pass

        if remove_stamps_file:
            logging.debug('Removing stamps file')
            os.remove(self._hdf5_stamps_fn)

    def create_stamp_slices(self, stamp_size=(10, 10), *args, **kwargs):
        """Create PANOPTES Stamp Cubes (PSC) for each point source.

        Creates a slice through the cube corresponding to a stamp and stores the
        subtracted data in the hdf5 table with key `stamp/<picid>`.

        Args:
            remove_cube (bool, optional): Remove the full cube from the hdf5 file after
                processing, defaults to False
            *args (TYPE): Description
            **kwargs (dict): `ipython_widget=True` can be passed to display progress
                within a notebook

        """

        logging.info("Starting stamps creation")
        errors = dict()

        skip_sources = list()

        for i, fn in tqdm(
                enumerate(self.files),
                total=self.num_frames,
                desc="Getting point sources".ljust(25)):
            logging.debug("Staring file: {}".format(fn))
            with fits.open(fn) as hdu:
                hdu_idx = 0
                if fn.endswith('.fz'):
                    hdu_idx = 1

                wcs = WCS(hdu[hdu_idx].header)
                d0 = hdu[hdu_idx].data

                try:
                    img_id = fn.split('/')[-1]
                    # Get the time from the image
                    img_id = img_id.split('.')[0]
                except ValueError as e:
                    logging.warning(e)
                    continue

            for star_row in self.point_sources.itertuples():
                star_id = str(star_row.Index)

                if star_id in skip_sources:
                    continue

                star_pos = wcs.all_world2pix(star_row.ra, star_row.dec, 0)

                try:
                    dset = self.stamps[star_id]
                except KeyError:
                    dset = self.stamps.create_dataset(
                        star_id + '/data',
                        (self.num_frames, stamp_size[0] * stamp_size[1]),
                        dtype='i2',
                        chunks=True
                    )

                try:
                    s0 = helpers.get_stamp_slice(star_pos[0], star_pos[1], stamp_size=stamp_size)
                    d1 = d0[s0].flatten()

                    if len(d1) == 0:
                        logging.debug('Bad slice for {}, skipping'.format(star_id))
                        skip_sources.append(star_id)
                        dset.attrs['quality'] = 'incomplete'
                        continue

                    dset[i] = d1

                    dset.attrs['picid'] = star_id
                    dset.attrs['ra'] = star_row.ra
                    dset.attrs['dec'] = star_row.dec
                    dset.attrs['twomass'] = star_row.twomass
                    dset.attrs['x'] = star_row.X
                    dset.attrs['y'] = star_row.Y
                    dset.attrs['wcs'] = str(wcs)
                    dset.attrs['seq_time'] = self.seq_time
                    dset.attrs['img_time'] = img_id
                except Exception as e:
                    if str(e) not in errors:
                        logging.warning("Error 01")
                        logging.warning(e)
                        errors[str(e)] = True
                finally:
                    self.stamps.flush()

    def find_similar_stars(self, target_index, store=True, force_new=True, *args, **kwargs):
        """ Get all variances for given target

        Args:
            stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
            i(int): Index of target PIC
        """
        num_sources = len(list(self.stamps.keys()))

        # Assume no match, i.e. high (99) value
        data = np.ones((num_sources)) * 99.

        if force_new:
            try:
                del self.hdf5_stamps['vgrid']
            except Exception as e:
                pass

        try:
            vgrid_dset = self.hdf5_stamps['vgrid']
        except KeyError:
            vgrid_dset = self.hdf5_stamps.create_dataset('vgrid', data=data)

        psc0 = self.get_psc(target_index)
        num_frames = psc0.shape[0]

        # Normalize
        logging.debug("Normalizing target for {} frames".format(num_frames))
        frames = []
        normalized_psc0 = np.zeros_like(psc0, dtype='f4')
        for frame_index in range(num_frames):
            try:
                if psc0[frame_index].sum() > 0.:
                    normalized_psc0[frame_index] = psc0[frame_index] / psc0[frame_index].sum()
                    frames.append(frame_index)
                else:
                    logging.debug("Sum for target frame {} is 0".format(frame_index))
            except RuntimeWarning:
                warn("Skipping frame {}".format(frame_index))

        iterator = list(self.stamps.keys())

        for i, source_index in tqdm(enumerate(iterator)):
            try:
                psc1 = self.get_psc(source_index)
            except Exception:
                continue

            normalized_psc1 = np.zeros_like(psc1, dtype='f4')

            # Normalize
            for frame_index in frames:
                if psc1[frame_index].sum() > 0.:
                    normalized_psc1[frame_index] = psc1[frame_index] / psc1[frame_index].sum()

            # Store in the grid
            try:
                v = ((normalized_psc0 - normalized_psc1) ** 2).sum()
                data[i] = v
                if store:
                    vgrid_dset[source_index] = v
            except ValueError as e:
                logging.debug("Skipping invalid stamp for source {}: {}".format(source_index, e))

        return data

    def get_stamp_collection(self, num_refs=25):

        vary = self.hdf5_stamps['vgrid']

        vary_series = pd.Series(vary)
        vary_series.sort_values(inplace=True)

        target_psc = self.get_psc(self.target_slice)

        num_frames = target_psc.data.shape[0]
        stamp_h = target_psc.data.shape[1]
        stamp_w = target_psc.data.shape[2]

        stamp_collection = np.array([self.get_psc(int(idx)).data for
                                     idx in vary_series.index[0:num_refs]])

        return stamp_collection.reshape(num_refs, num_frames, stamp_h * stamp_w)

    def get_ideal_full_coeffs(self, stamp_collection, damp=1, func=lsmr, verbose=False):

        num_frames = stamp_collection.shape[1]
        num_pixels = stamp_collection.shape[2]

        target_frames = stamp_collection[0].flatten()
        refs_frames = stamp_collection[1:].reshape(-1, num_frames * num_pixels).T

        if verbose:
            print("Target other shape: {}".format(target_frames.shape))
            print("Refs other shape: {}".format(refs_frames.shape))

        coeffs = func(refs_frames, target_frames, damp)

        return coeffs

    def get_ideal_full_psc(self, stamp_collection, coeffs, **kwargs):

        refs = stamp_collection[1:]

        created_frame = (refs.T * coeffs).sum(2).T

        return created_frame

    def get_stamp_mask(self, source_index, **kwargs):
        r_min, r_max, c_min, c_max = self.get_stamp_bounds(source_index, **kwargs)

        masks = []
        for mask in self.rgb_masks:
            masks.append(mask[r_min:r_max, c_min:c_max])

        return masks

    def lookup_point_sources(self,
                             image_num=0,
                             use_sextractor=True,
                             use_tess_catalog=False,
                             sextractor_params=None,
                             force_new=False
                             ):
        """ Extract point sources from image

        Args:
            image_num (int, optional): Frame number of observation from which to
                extract images
            sextractor_params (dict, optional): Parameters for sextractor,
                defaults to settings contained in the `panoptes.sex` file
            force_new (bool, optional): Force a new catalog to be created,
                defaults to False

        Raises:
            error.InvalidSystemCommand: Description
        """
        if use_sextractor:
            # Write the sextractor catalog to a file
            source_file = '{}/point_sources_{:02d}.cat'.format(self.data_dir, image_num)
            logging.debug("Point source catalog: {}".format(source_file))

            if not os.path.exists(source_file) or force_new:
                logging.debug("No catalog found, building from sextractor")
                # Build catalog of point sources
                sextractor = shutil.which('sextractor')
                if sextractor is None:
                    sextractor = shutil.which('sex')
                    if sextractor is None:
                        raise Exception('sextractor not found')

                if sextractor_params is None:
                    sextractor_params = [
                        '-c', '{}/PIAA/resources/conf_files/sextractor/panoptes.sex'.format(
                            os.getenv('PANDIR')),
                        '-CATALOG_NAME', source_file,
                    ]

                logging.debug("Running sextractor...")
                cmd = [sextractor, *sextractor_params, self.files[image_num]]
                logging.debug(cmd)
                subprocess.run(cmd)

            # Read catalog
            point_sources = Table.read(source_file, format='ascii.sextractor')

            # Remove the point sources that sextractor has flagged
            if 'FLAGS' in point_sources.keys():
                point_sources = point_sources[point_sources['FLAGS'] == 0]
                point_sources.remove_columns(['FLAGS'])

            # Rename columns
            point_sources.rename_column('X_IMAGE', 'X')
            point_sources.rename_column('Y_IMAGE', 'Y')

            # Add the SNR
            point_sources['SNR'] = point_sources['FLUX_AUTO'] / point_sources['FLUXERR_AUTO']

            # Filter point sources near edge
            # w, h = data[0].shape
            w, h = (3476, 5208)

            stamp_size = 60

            top = point_sources['Y'] > stamp_size
            bottom = point_sources['Y'] < w - stamp_size
            left = point_sources['X'] > stamp_size
            right = point_sources['X'] < h - stamp_size

            self._point_sources = point_sources[top & bottom & right & left].to_pandas()
            ra_key = 'ALPHA_J2000'
            dec_key = 'DELTA_J2000'

        if use_tess_catalog:
            wcs_footprint = self.wcs.calc_footprint()
            ra_max = max(wcs_footprint[:, 0])
            ra_min = min(wcs_footprint[:, 0])
            dec_max = max(wcs_footprint[:, 1])
            dec_min = min(wcs_footprint[:, 1])
            logging.debug("RA: {:.03f} - {:.03f} \t Dec: {:.03f} - {:.03f}".format(ra_min,
                                                                                   ra_max,
                                                                                   dec_min,
                                                                                   dec_max))
            self._point_sources = helpers.get_stars(
                ra_min, ra_max, dec_min, dec_max, cursor_only=False)

            star_pixels = self.wcs.all_world2pix(
                self._point_sources['ra'], self._point_sources['dec'], 0)
            self._point_sources['X'] = star_pixels[0]
            self._point_sources['Y'] = star_pixels[1]

            self._point_sources.add_index(['id'])
            self._point_sources = self._point_sources.to_pandas()
            ra_key = 'ra'
            dec_key = 'dec'

        # Do catalog matching
        stars = SkyCoord(ra=self._point_sources[ra_key].values *
                         u.deg, dec=self._point_sources[dec_key].values * u.deg)
        st0 = helpers.get_stars_from_footprint(self.wcs.calc_footprint(), cursor_only=False)
        catalog = SkyCoord(ra=st0['ra'] * u.deg, dec=st0['dec'] * u.deg)
        idx, d2d, d3d = match_coordinates_sky(stars, catalog)

        self._point_sources['id'] = st0[idx]['id']
        self._point_sources.set_index('id', inplace=True)

    def _load_images(self):
        seq_files = sorted(glob("{}/*T*.fits*".format(self.data_dir)))

        self.files = seq_files
        if len(self.files) > 0:
            self._num_frames = len(self.files)
