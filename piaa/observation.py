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
from astropy.nddata.utils import Cutout2D, PartialOverlapError, NoOverlapError
from astropy.time import Time

from scipy.sparse.linalg import lsmr
from scipy import linalg

from dateutil.parser import parse as date_parse

from tqdm import tqdm

from piaa.utils import make_masks
from piaa.utils import helpers
from piaa import exoplanets

from pocs.utils.images import fits as fits_utils

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

        log_file = '/var/panoptes/logs/{}.log'.format(sequence.replace('/', '_'))
        logging.basicConfig(
            filename=log_file, 
            level=logging.INFO, 
            format='%(asctime)s %(message)s'
        )
        
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
            logging.info("Creating stamps file.")
            stamp_group = self.hdf5_stamps.create_group('stamps')

            # Attach image times to group
            logging.info("Adding image times to stamps file")
            stamp_group.create_dataset('image_times', data=self.image_times)
            self.hdf5_stamps.flush()

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
            image_times = list()

            try:
                for i, fn in enumerate(self.files):
                    # Get the time from the image
                    try:
                        date_obs = self.get_header_value(i, 'IMAGEID').split('_')[-1]
                        t0 = Time(date_parse(date_obs))
                    except Exception as e:
                        logging.warning(e)
                        continue

                    image_times.append(t0.mjd)
            except AttributeError: # Don't have files
                image_times = Time(self.stamps['image_times'], format='mjd')

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
                logging.debug("Making RGB masks - {}".format(rgb_mask_file))
                self._rgb_masks = np.array(make_masks(self.data_cube[0]))
                self._rgb_masks.dump(rgb_mask_file)

        return self._rgb_masks

    @property
    def transit_info(self, picid):
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

    def get_psc(self, picid, frame_slice=None, subtract_bias=2048):
        try:
            psc = np.array(self.stamps[picid]['data']) - subtract_bias
        except KeyError:
            raise Exception("{} not found in the stamp collection.".format(picid))

        if frame_slice is not None:
            psc = psc[frame_slice]

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
        logging.info('Creating stamps for {}'.format(self.sequence))

        if not force_new:
            # Check for file locally
            if os.path.exists(self._hdf5_stamps_fn):
                logging.info('Using local stamps file')
                return

            # Check if in storage bucket
            stamp_blob = helpers.get_observation_blobs(key=self.sequence + '.hdf5')
            if stamp_blob:
                logging.info('Downloading stamps file from storage bucket')
                helpers.download_blob(stamp_blob, save_as=self._hdf5_stamps_fn)
                return

        # Download FITS files
        logging.info('Downloading FITS files')
        fits_blobs = helpers.get_observation_blobs(self.sequence)
        
        # Skip short observations
        if len(fits_blobs) < 20:
            logging.info('Skipping short observations')
            raise Exception('Skipping short observations')

        # Download all the FITS files from a bucket
        self.files = list()
        if fits_blobs:
            for blob in tqdm(fits_blobs, desc='Downloading FITS files'.ljust(25)):
                fits_fn = helpers.unpack_blob(blob, save_dir=self._data_dir)
                self.files.append(fits_fn)

        self.num_frames = len(self.files)

        # Plate-solve all the images - safe to run again
        logging.info('Plate-solving FITS files')
        solved_files = list()
        for fn in tqdm(self.files, desc='Solving files'.ljust(25)):
            try:
                fits_utils.get_solve_field(fn, timeout=90)
                solved_files.append(fn)
            except Exception:
                logging.warning("Can't solve file {}".format(fn))
                logging.debug("Stopping processing for sequence and cleaning up")
                continue
                #self._do_cleanup(remove_stamps_file=True)
                
        self.files = solved_files

        # Lookup point sources
        # You need to set the env variable for the password for TESS catalog DB (ask Wilfred)
        # os.environ['PGPASSWORD'] = 'sup3rs3cr3t'
        logging.info('Looking up point sources via TESS catalog')
        self.lookup_point_sources(use_sextractor=False, use_tess_catalog=True, **kwargs)
        logging.info("Number of sources detected: {}".format(len(self.point_sources)))

        # Create stamps
        logging.info('Creating stamps')
        self.create_stamp_slices(stamp_size=stamp_size)

        # Upload to storage bucket
        if upload:
            logging.info('Uploading stamps file to storage bucket')
            helpers.upload_to_bucket(self._hdf5_stamps_fn, self.sequence + '.hdf5')

        # Cleanup
        if cleanup_after:
            self._do_cleanup(remove_stamps_file=remove_stamps_file)

    def _do_cleanup(self, remove_stamps_file=False):
        logging.info('Cleaning up FITS files')
        for fn in self.files:
            os.remove(fn)
            try:
                os.remove(fn.replace('.fits', '.solved'))
            except Exception:
                pass

        if remove_stamps_file:
            logging.info('Removing stamps file')
            os.remove(self._hdf5_stamps_fn)

    def create_stamp_slices(self, stamp_size=(10, 10), snr_limit=5, *args, **kwargs):
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
            logging.debug("Starting file: {}".format(fn))
            with fits.open(fn) as hdu:
                hdu_idx = 0
                if fn.endswith('.fz'):
                    logging.debug("Using compressed FITS")
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
                    
            high_snr = self.point_sources.snr > snr_limit
            sources = self.point_sources[high_snr]

            logging.info("Looping through point sources: {}/{}".format(i, len(self.files)))
            for star_row in sources.itertuples():
                star_id = str(star_row.Index)

                if star_id in skip_sources:
                    continue

                star_pos = wcs.all_world2pix(star_row.ra, star_row.dec, 0)

                # Get stamp data. If problem, mark for skipping in future.
                try:
                    s0 = helpers.get_stamp_slice(star_pos[0], star_pos[1], stamp_size=stamp_size)
                    d1 = d0[s0].flatten()

                    if len(d1) == 0:
                        logging.warning('Bad slice for {}, skipping'.format(star_id))
                        skip_sources.append(star_id)
                        continue
                except Exception as e:
                    raise e

                # Get or create the group to hold the PSC
                try:
                    psc_group = self.stamps[star_id]
                except KeyError:
                    logging.debug("Creating new group for star {}".format(star_id))
                    psc_group = self.stamps.create_group(star_id)
                    # Stamp metadata
                    try:
                        psc_metadata = {
                            'ra': star_row.ra,
                            'dec': star_row.dec,
                            'twomass': star_row.twomass,
                            'vmag': star_row.vmag,
                            'tmag': star_row.tmag,
                            'seq_time': self.seq_time,
                        }
                        for k, v in psc_metadata.items():
                            psc_group.attrs[k] = str(v)
                    except Exception as e:
                        if str(e) not in errors:
                            logging.warning(e)
                            errors[str(e)] = True

                # Set the data for the stamp. Create PSC dataset if needed.
                try:
                    # Assign stamp values
                    psc_group['data'][i] = d1
                except KeyError:
                    logging.debug("Creating new PSC dataset for {}".format(star_id))
                    psc_size = (self.num_frames, len(d1))

                    # Create the dataset
                    stamp_dset = psc_group.create_dataset(
                            'data', psc_size, dtype='u2', chunks=True)

                    # Assign the data
                    stamp_dset[i] = d1
                except TypeError as e:
                # Sets the metadata. Create metadata dataset if needed.
                    key = str(e) + star_id
                    if key not in errors:
                        logging.debug(e)
                        errors[key] = True

                try:
                    psc_group['original_position'][i] = (star_pos[0], star_pos[1])
                except KeyError:
                    logging.debug("Creating new metadata dataset for {}".format(star_id))
                    metadata_size = (self.num_frames, 2)

                    # Create the dataset
                    metadata_dset = psc_group.create_dataset(
                            'original_position', metadata_size, dtype='u2', chunks=True)

                    # Assign the data
                    metadata_dset[i] = (star_row.x, star_row.y)
                finally:
                    self.hdf5_stamps.flush()

    def find_similar_stars(self, target_index, store=True, force_new=False, *args, **kwargs):
        """ Get all variances for given target

        Args:
            stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
            i(int): Index of target PIC
        """
        vary_fn = '/var/panoptes/images/lc/{}_{}.csv'.format(self.sequence.replace('/','_'), target_index)
        if force_new:
            try:
                logging.debug("Removing exsting comparison stars and forcing new")
                os.remove(vary_fn)
            except FileNotFoundError as e:
                pass
            
        try:
            return pd.read_csv(vary_fn, index_col=[0])
        except Exception:
            logging.debug("Can't find stored similar stars, generating new")
            
        num_sources = len(list(self.stamps.keys()))

        # Assume no match, i.e. high (99) value
        data = dict()
    
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

        for i, source_index in tqdm(enumerate(iterator), desc="Finding similar sources", total=len(self.stamps)):
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
                data[source_index] = v
            except ValueError as e:
                logging.debug("Skipping invalid stamp for source {}: {}".format(source_index, e))
                
        df0 = pd.DataFrame(
                {'v': list(data.values())}, 
                index=list(data.keys())).sort_values(by='v')
        
        if store:
            df0[0:500].to_csv(vary_fn)

        return df0

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

    def get_ideal_full_coeffs(self, stamp_collection, verbose=False):

        num_frames = stamp_collection.shape[1]
        num_pixels = stamp_collection.shape[2]

        target_frames = stamp_collection[0].flatten()
        refs_frames = stamp_collection[1:].reshape(-1, num_frames * num_pixels).T
        
        if verbose:
            print("Stamp collection shape: {}".format(stamp_collection.shape))
            print("Target shape: {}".format(target_frames.shape))
            print("Refs shape: {}".format(refs_frames.shape))

        coeffs = linalg.lstsq(refs_frames, target_frames)

        return coeffs

    def get_ideal_full_psc(self, stamp_collection, coeffs, **kwargs):

        refs = stamp_collection[1:]

        created_frame = (refs.T * coeffs).sum(2).T
        #print("USING MEAN FOR COMPARISON")
        #created_frame = refs.mean(0)

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
                             force_new=False,
                             **kwargs
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
            self._point_sources.columns = [
                'x', 'y', 
                'ra', 'dec', 
                'background', 
                'flux_auto', 'flux_max', 'fluxerr_auto', 
                'fwhm', 'snr'
            ]

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
                ra_min, ra_max, dec_min, dec_max, cursor_only=False, table=kwargs.get('table', 'full_catalog'))

            star_pixels = self.wcs.all_world2pix(
                self._point_sources['ra'], self._point_sources['dec'], 0)
            self._point_sources['X'] = star_pixels[0]
            self._point_sources['Y'] = star_pixels[1]

            self._point_sources.add_index(['id'])
            self._point_sources = self._point_sources.to_pandas()

        # Do catalog matching
        stars = SkyCoord(ra=self._point_sources['ra'].values *
                         u.deg, dec=self._point_sources['dec'].values * u.deg)
        st0 = helpers.get_stars_from_footprint(self.wcs.calc_footprint(), cursor_only=False, table=kwargs.get('table', 'full_catalog'))
        catalog = SkyCoord(ra=st0['ra'] * u.deg, dec=st0['dec'] * u.deg)
        idx, d2d, d3d = match_coordinates_sky(stars, catalog)

        self._point_sources['id'] = st0[idx]['id']
        self._point_sources['twomass'] = st0[idx]['twomass']
        self._point_sources['tmag'] = st0[idx]['tmag']
        self._point_sources['vmag'] = st0[idx]['vmag']
        self._point_sources.set_index('id', inplace=True)
        
        return d2d
        
    def normalize(self, cube):
        return (cube.T / cube.sum(1)).T

    def run_piaa(self, picid, num_refs=50, aperture_size=6, exp_time=120, save_csv=True, force_new=False, new_comparisons=False, show_stamps=False, verbose=False):
        try:
            diff_group = self.stamps['diffs']
        except KeyError:
            diff_group = self.stamps.create_group('diffs')

        if force_new:
            try:
                del diff_group[picid]
            except KeyError:
                pass
            
        temp_psc = self.get_psc(str(picid))
        if len(temp_psc) < 20:
            raise Exception("Not enough frames for sequence.")
            
        logging.info("Running the PIAA reduction")
        vary_series = self.find_similar_stars(picid, force_new=new_comparisons, display_progress=False, store=True)

        logging.info("Building collection")
        #ref_collection = np.array([self.get_psc(str(idx)) for idx in vary_series.index[:num_refs]])
        ref_collection = np.array([self.get_psc(str(idx)) for idx in vary_series.index[:num_refs:3]])
        self.num_frames = ref_collection[0].shape[0]
        logging.info("Ref collection shape: {}".format(ref_collection.shape))

        # Normalize each PSC
        logging.info("Normalizing collection")
        normalized_collection = np.array([self.normalize(s) for s in ref_collection])
        logging.info("Normalized ref collection shape: {}".format(normalized_collection.shape))
        logging.info(normalized_collection.sum(2))

        # Build the coeffs off the normalized PSC
        logging.info("Getting coefficients: num_refs={} aperture={}".format(num_refs, aperture_size))
        coeffs = self.get_ideal_full_coeffs(normalized_collection, verbose=verbose)
        logging.info(coeffs)
        logging.debug(normalized_collection)

        # Build the template from the coeffs with non-normalized data
        logging.info("Building ideal stamp")
        ideal = self.get_ideal_full_psc(ref_collection, coeffs[0], verbose=verbose).reshape(self.num_frames, -1)

        target_psc = ref_collection[0]
        
        if picid in diff_group:
            logging.info('Results exists from previous')
            return target_psc, ideal
        
        stamp_side = int(np.sqrt(target_psc.shape[1]))
        stamp_size = (stamp_side, stamp_side)

        rgb_stamp_masks = helpers.get_rgb_masks(target_psc[0].reshape(stamp_size[0], stamp_size[1]), force_new=True, separate_green=False)
        

        diff = list()
        for frame_idx in range(self.num_frames):
            d0 = target_psc[frame_idx].reshape(stamp_size[0], stamp_size[1])
            i0 = ideal[frame_idx].reshape(stamp_size[0], stamp_size[1])

            star_pos = np.array(self.stamps[picid]['original_position'])[frame_idx]
            slice0 = helpers.get_stamp_slice(star_pos[0], star_pos[1], stamp_size=stamp_size)

            try:
                #aperture_position = (star_pos[0] - slice0[1].start, star_pos[1] - slice0[0].start)
                #aperture_position = (star_pos[1] - slice0[0].start, star_pos[0] - slice0[1].start)
                y_pos, x_pos = np.argwhere(d0 == d0.max())[0]
                aperture_position = (x_pos, y_pos)
            except IndexError:
                logging.warning("No star position: ", frame_idx, slice0, star_pos)
                aperture_position = (5, 5)
                
            logging.info("Getting cutout at {} of size {}".format(aperture_position, aperture_size))
            
            color_flux = list()
            stamps = list()
            for color, mask in zip('rgb', rgb_stamp_masks):
                
                d1 = np.ma.array(d0, mask=~mask)
                i1 = np.ma.array(i0, mask=~mask)
            
                try:
                    d2 = Cutout2D(d1, aperture_position, aperture_size, mode='strict')
                    i2 = Cutout2D(i1, aperture_position, aperture_size, mode='strict')
                except (PartialOverlapError, NoOverlapError) as e:
                    logging.warning("Bad overlap frame {}, color {}".format(frame_idx, color))
                    logging.warning(aperture_position)
                    continue
                except Exception as e:
                    logging.warning("Prolem with cutout: {}".format(e))
                    continue

                d3 = d2.data
                i3 = i2.data

                color_flux.append(d3.sum())
                color_flux.append(i3.sum())
                
                stamps.append([d3, i3])
                
            if show_stamps:
                target_s = stamps[0][0].filled(0) + stamps[1][0].filled(0) + stamps[2][0].filled(0)
                ideal_s = stamps[0][1].filled(0) + stamps[1][1].filled(0) + stamps[2][1].filled(0)
                helpers.show_stamps(
                    [target_s,ideal_s], 
                    stamp_size=aperture_size, 
                    save_name='/var/panoptes/images/lc/stamps/{}_{}_{}_{:02d}.png'.format(
                        self.cam_id, picid, self.seq_time, frame_idx
                    )
                )
                
            diff.append(color_flux)
            
        diffs = np.array(diff)
        logging.info(diffs.shape)
        logging.info(diffs)
        image_times = Time(self.stamps['image_times'], format='mjd')

        try:
            lc = pd.DataFrame(diffs, index=image_times, columns=[
                'r_target', 'r_ideal',
                'g_target', 'g_ideal',
                'b_target', 'b_ideal',
            ])            

            csv_file = '/var/panoptes/images/lc/{}_{}_diff.csv'.format(self.sequence.replace('/', '_'), picid)
            logging.info("Writing csv to {}".format(csv_file))
            lc.to_csv(csv_file)
        except Exception as e:
            logging.warning("Problem createing CSV file: {}".format(e))

        return target_psc, ideal

    def _load_images(self):
        seq_files = sorted(glob("{}/*T*.fits*".format(self.data_dir)))

        self.files = seq_files
        if len(self.files) > 0:
            self._num_frames = len(self.files)
