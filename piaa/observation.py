import logging
import os
import shutil
import subprocess

from collections import namedtuple
from glob import glob

from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.utils.console import ProgressBar
from astropy.wcs import WCS

from photutils import RectangularAperture
from photutils import aperture_photometry
from photutils import make_source_mask

import h5py
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from . import utils
from pocs.utils import error


Stamp = namedtuple('Stamp', ['row_slice', 'col_slice', 'mid_point'])


class Observation(object):

    def __init__(self, image_dir, aperture_size=6, camera_bias=1024):
        """ A sequence of images to be processed as one observation """
        assert os.path.exists(image_dir), "Specified directory does not exist"

        logging.basicConfig(filename='/var/panoptes/logs/piaa.log', level=logging.DEBUG)
        self.logger = logging
        self.logger.info('Setting up Observation for analysis')

        super(Observation, self).__init__()

        if image_dir.endswith('/'):
            image_dir = image_dir[:-1]
        self._image_dir = image_dir

        self.camera_bias = camera_bias

        self._img_h = 3476
        self._img_w = 5208
        # Background regions
        self._back_h = int(self._img_h // 5)
        self._back_w = int(self._img_w // 5)

        self.background_region = {}

        self.aperture_size = aperture_size
        self.stamp_size = (None, None)

        self._point_sources = None
        self._pixel_locations = None

        self._stamp_masks = (None, None, None)
        self._stamps_cache = {}

        self._hdf5 = h5py.File(image_dir + '.hdf5')
        self._hdf5_normalized = h5py.File(image_dir + '_normalized.hdf5')

        self._load_images()

    @property
    def point_sources(self):
        if self._point_sources is None:
            self.lookup_point_sources()

        return self._point_sources

    @property
    def image_dir(self):
        """ Image directory containing FITS files

        When setting a new image directory, FITS files are automatically
        loaded into the `files` property

        Returns:
            str: Path to image directory
        """
        return self._image_dir

    @image_dir.setter
    def image_dir(self, directory):
        self._load_images()

    @property
    def pixel_locations(self):
        if self._pixel_locations is None:
            # Get RA/Dec coordinates from first frame
            ra = self.point_sources['ALPHA_J2000']
            dec = self.point_sources['DELTA_J2000']

            locs = list()

            for f in self.files:
                wcs = WCS(f)
                xy = np.array(wcs.all_world2pix(ra, dec, 1, ra_dec_order=True))

                # Transpose
                locs.append(xy.T)

            locs = np.array(locs)
            self._pixel_locations = pd.Panel(locs)

        return self._pixel_locations

    @property
    def stamps(self):
        return self._stamps_cache

    # @property
    # def num_frames(self):
    #     assert self.psc_collection is not None
    #     return self.psc_collection.shape[0]

    # @property
    # def num_stars(self):
    #     assert self.psc_collection is not None
    #     return self.psc_collection.shape[1]

    @property
    def data_cube(self):
        try:
            cube_dset = self._hdf5['cube']
            self.logger.debug("Getting existing data cube")
        except KeyError:
            self.logger.debug("Creating data cube")
            cube_dset = self._hdf5.create_dataset('cube', (len(self.files), self._img_h, self._img_w))
            for i, f in enumerate(self.files):
                cube_dset[i] = fits.getdata(f)

        return cube_dset

    def subtract_background(self, stamp,
                            r_mask=None, g_mask=None, b_mask=None, mid_point=None, frame_index=None,
                            background_sub_method='median'):
        """ Perform RGB background subtraction

        Args:
            stamp (numpy.array): A stamp of the data
            r_mask (numpy.ma.array, optional): A mask of the R channel
            g_mask (numpy.ma.array, optional): A mask of the G channel
            b_mask (numpy.ma.array, optional): A mask of the B channel
            background_sub_method (str, optional): Subtraction method of `median` or `mean`

        Returns:
            numpy.array: The background subtracted data recomined into one array
        """
        # self.logger.debug("Subtracting background - {}".format(background_sub_method))

        background_region_id = (int(mid_point[1] // self._back_w), int(mid_point[0] // self._back_h))
        self.logger.debug("Background region: {}\tFrame: {}".format(background_region_id, frame_index))

        try:
            frame_background = self.background_region[frame_index]
        except KeyError:
            frame_background = dict()
            self.background_region[frame_index] = frame_background

        try:
            background_region = frame_background[background_region_id]
        except KeyError:
            background_region = dict()
            self.background_region[frame_index][background_region_id] = background_region

        try:
            r_channel_background = background_region['red']
            g_channel_background = background_region['green']
            b_channel_background = background_region['blue']
        except KeyError:
            r_channel_background = list()
            g_channel_background = list()
            b_channel_background = list()
            self.background_region[frame_index][background_region_id]['red'] = r_channel_background
            self.background_region[frame_index][background_region_id]['green'] = g_channel_background
            self.background_region[frame_index][background_region_id]['blue'] = b_channel_background

        self.logger.debug("R channel background {}".format(r_channel_background))
        self.logger.debug("G channel background {}".format(g_channel_background))
        self.logger.debug("B channel background {}".format(b_channel_background))

        if len(r_channel_background) < 5:

            self.logger.debug("Getting source mask {} {} {}".format(type(stamp), stamp.dtype, stamp.shape))
            source_mask = make_source_mask(stamp, snr=3., npixels=2)
            self.logger.debug("Got source mask")

            if r_mask is None or g_mask is None or b_mask is None:
                self.logger.debug("Making RGB masks for data subtraction")
                self._stamp_masks = utils.make_masks(stamp)
                r_mask, g_mask, b_mask = self._stamp_masks

            method_lookup = {
                'mean': 0,
                'median': 1,
            }
            method_idx = method_lookup[background_sub_method]

            self.logger.debug("Determining backgrounds")
            r_masked_data = np.ma.array(stamp, mask=np.logical_or(source_mask, ~r_mask))
            r_stats = sigma_clipped_stats(r_masked_data, sigma=3.)
            r_back = r_stats[method_idx]
            r_channel_background.append(r_back)

            g_masked_data = np.ma.array(stamp, mask=np.logical_or(source_mask, ~g_mask))
            g_stats = sigma_clipped_stats(g_masked_data, sigma=3.)
            g_back = g_stats[method_idx]
            g_channel_background.append(g_back)

            b_masked_data = np.ma.array(stamp, mask=np.logical_or(source_mask, ~b_mask))
            b_stats = sigma_clipped_stats(b_masked_data, sigma=3.)
            b_back = b_stats[method_idx]
            b_channel_background.append(b_back)

            # Store the background values
            self.logger.debug("Storing new background values: {} {} {}".format(
                r_channel_background, g_channel_background, b_channel_background))
            self.background_region[frame_index][background_region_id]['red'] = r_channel_background
            self.background_region[frame_index][background_region_id]['green'] = g_channel_background
            self.background_region[frame_index][background_region_id]['blue'] = b_channel_background
            self.logger.debug("Values stored")

        # Use average of others
        self.logger.debug("Subtracting mean backgrounds")
        r_back = np.median(r_channel_background)
        g_back = np.median(g_channel_background)
        b_back = np.median(b_channel_background)

        self.logger.debug("Background subtraction: Region {} {}\t{}\t{}".format(
            background_region_id, r_back, g_back, b_back))
        r_masked_data = np.ma.array(stamp, mask=~r_mask) - int(r_back)
        g_masked_data = np.ma.array(stamp, mask=~g_mask) - int(g_back)
        b_masked_data = np.ma.array(stamp, mask=~b_mask) - int(b_back)

        # self.logger.debug("Combining channels")
        subtracted_data = r_masked_data.filled(0) + g_masked_data.filled(0) + b_masked_data.filled(0)

        # self.logger.debug("Removing saturated")
        subtracted_data[subtracted_data > 1e4] = 1e-5

        return subtracted_data

    def get_source_slice(self, source_index, force_new=False, cache=True, *args, **kwargs):
        """ Create a stamp (stamp) of the data

        This uses the start and end points from the source drift to figure out
        an appropriate size to stamp. Data is bias and background subtracted.
        """
        try:
            if force_new:
                del self._stamps_cache[source_index]
            stamp = self._stamps_cache[source_index]
        except KeyError:
            start_pos, mid_pos, end_pos = self._get_stamp_points(source_index)
            mid_pos = self._adjust_stamp_midpoint(mid_pos)

            # Get the width and height of data region
            width, height = (start_pos - end_pos)

            cutout = Cutout2D(
                fits.getdata(self.files[0]),
                (mid_pos[0], mid_pos[1]),
                (self._nearest_10(height) + 8, self._nearest_10(width) + 4)
            )

            xs, ys = cutout.bbox_original

            stamp = Stamp(slice(xs[0], xs[1] + 1), slice(ys[0], ys[1] + 1), mid_pos)

            if cache:
                self._stamps_cache[source_index] = stamp

            # Shared across all stamps
            self.stamp_size = cutout.data.shape

        return stamp

    def get_source_fluxes(self, source_index):
        """ Get fluxes for given source

        Args:
            source_index (int): Index of the source from `point_sources`

        Returns:
            numpy.array: 1-D array of fluxes
        """
        fluxes = []

        stamps = self.get_source_stamps(source_index)

        # Get aperture photometry
        for i in self.pixel_locations:
            x = int(self.pixel_locations[i, source_index, 0] - stamps[i].origin_original[0]) - 0.5
            y = int(self.pixel_locations[i, source_index, 1] - stamps[i].origin_original[1]) - 0.5

            aperture = RectangularAperture((x, y), w=6, h=6, theta=0)

            phot_table = aperture_photometry(stamps[i].data, aperture)

            flux = phot_table['aperture_sum'][0]

            fluxes.append(flux)

        fluxes = np.array(fluxes)

        return fluxes

    def get_frame_stamp(self, source_index, frame_index, *args, **kwargs):
        """ Get individual stamp for given source and frame

        Note:
            Data is bias and background subtracted

        Args:
            source_index (int): Index of the source from `point_sources`
            frame_index (int): Index of the frame from `files`
            *args (TYPE): Description
            **kwargs (TYPE): Description

        Returns:
            numpy.array: Array of data
        """
        stamps = self.get_source_stamps(source_index, *args, **kwargs)

        stamp = stamps[frame_index]

        return stamp

    def get_frame_aperture(self, source_index, frame_index, width=6, height=6, *args, **kwargs):
        """Aperture for given frame from source

        Note:
            `width` and `height` should be in multiples of 2 to get a super-pixel

        Args:
            source_index (int): Index of the source from `point_sources`
            frame_index (int): Index of the frame from `files`
            width (int, optional): Width of the aperture, defaults to 3x2=6
            height (int, optional): Height of the aperture, defaults to 3x2=6
            *args (TYPE): Description
            **kwargs (TYPE): Description

        Returns:
            photutils.RectangularAperture: Aperture surrounding the frame
        """
        stamp = self.get_frame_stamp(source_index, frame_index, *args, **kwargs)

        x = int(self.pixel_locations[frame_index, source_index, 0] - stamp.origin_original[0]) - 0.5
        y = int(self.pixel_locations[frame_index, source_index, 1] - stamp.origin_original[1]) - 0.5

        aperture = RectangularAperture((x, y), w=width, h=height, theta=0)

        return aperture

    def plot_stamp(self, source_index, frame_index, show_bayer=True, show_data=False, *args, **kwargs):

        stamp = self.get_frame_stamp(source_index, frame_index, *args, **kwargs)

        fig, (ax1) = plt.subplots(1, 1, facecolor='white')
        fig.set_size_inches(30, 15)

        aperture = self.get_frame_aperture(source_index, frame_index, return_aperture=True)

        aperture_mask = aperture.to_mask(method='center')[0]
        aperture_data = aperture_mask.cutout(stamp.data)

        phot_table = aperture_photometry(stamp.data, aperture, method='center')

        if show_data:
            print(np.flipud(aperture_data))  # Flip the data to match plot

        cax = ax1.imshow(stamp.data, cmap='cubehelix_r', vmin=0., vmax=aperture_data.max())
        fig.colorbar(cax)

        aperture.plot(color='b', ls='--', lw=2)

        if show_bayer:
            # Bayer pattern
            for i, val in np.ndenumerate(stamp.data):
                x, y = stamp.to_original_position((i[1], i[0]))
                ax1.text(x=i[1], y=i[0], ha='center', va='center',
                         s=utils.pixel_color(x, y, zero_based=True), fontsize=10, alpha=0.25)

            # major ticks every 2, minor ticks every 1
            x_major_ticks = np.arange(-0.5, stamp.bbox_cutout[1][1], 2)
            x_minor_ticks = np.arange(-0.5, stamp.bbox_cutout[1][1], 1)

            y_major_ticks = np.arange(-0.5, stamp.bbox_cutout[0][1], 2)
            y_minor_ticks = np.arange(-0.5, stamp.bbox_cutout[0][1], 1)

            ax1.set_xticks(x_major_ticks)
            ax1.set_xticks(x_minor_ticks, minor=True)
            ax1.set_yticks(y_major_ticks)
            ax1.set_yticks(y_minor_ticks, minor=True)

            ax1.grid(which='major', color='r', linestyle='-', alpha=0.25)
            ax1.grid(which='minor', color='r', linestyle='-', alpha=0.1)

        ax1.set_title("Source {} Frame {} Aperture Flux: {}".format(source_index,
                                                                    frame_index, phot_table['aperture_sum'][0]))

        return fig

    def create_normalized_stamps(self, remove_cube=False, *args, **kwargs):
        """Create normalized stamps for entire data cube

        Creates a slice through the cube corresponding to a stamp and stores the
        normalized data in the hdf5 table with key `normalized/<index>`, where
        `<index>` is the source index from `point_sources`

        Args:
            remove_cube (bool, optional): Remove the full cube from the hdf5 file after
                processing, defaults to False
            *args (TYPE): Description
            **kwargs (dict): `ipython_widget=True` can be passed to display progress
                within a notebook

        """

        r_mask = None
        g_mask = None
        b_mask = None

        for source_index in ProgressBar(self.point_sources.index,
                                        ipython_widget=kwargs.get('ipython_widget', False)):

            group_name = 'normalized/{}'.format(source_index)
            if group_name not in self._hdf5_normalized:

                try:
                    ss = self.get_source_slice(source_index)
                    stamp = np.array(self.data_cube[:, ss.row_slice, ss.col_slice])

                    if r_mask is None:
                        r_mask, g_mask, b_mask = utils.make_masks(stamp[0])

                    self.logger.debug("Performing bias subtraction")
                    # Bias and background subtraction
                    stamp -= self.camera_bias

                    self.logger.debug("Performing background subtraction: Source {}".format(source_index))
                    stamps_clean = list()
                    for i, s in enumerate(stamp):
                        try:
                            stamps_clean.append(
                                self.subtract_background(s, r_mask=r_mask, g_mask=g_mask, b_mask=b_mask,
                                                         mid_point=ss.mid_point, frame_index=i))
                        except Exception as e:
                            self.logger.warning(
                                "Problem subtracting background for stamp {} frame {}: {}".format(source_index, i, e))

                    stamps_clean = np.array(stamps_clean)

                    # Normalize
                    self.logger.debug("Normalizing")
                    stamps_clean /= stamps_clean.sum()

                    # Store
                    self._hdf5_normalized.create_dataset(group_name, data=stamp)
                except Exception as e:
                    self.logger.warning("Problem creating normalized stamp for {}: {}".format(source_index, e))

    def get_variance_for_target(self, target_index, *args, **kwargs):
        """ Get all variances for given target

        Args:
            stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
            i(int): Index of target PIC
        """
        num_sources = len(self.point_sources)

        v = np.zeros((num_sources), dtype=np.float)

        stamp0 = np.array(self._hdf5['normalized/{}'.format(target_index)])

        for source_index in ProgressBar(range(num_sources), ipython_widget=kwargs.get('ipython_widget', False)):
            stamp1 = np.array(self._hdf5['normalized/{}'.format(source_index)])

            v[source_index] = ((stamp0 - stamp1) ** 2).sum()

        return v

    def lookup_point_sources(self, image_num=0, sextractor_params=None, force_new=False):
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
        # Write the sextractor catalog to a file
        source_file = '{}/test{:02d}.cat'.format(self.image_dir, image_num)
        self.logger.debug("Point source catalog: {}".format(source_file))

        if not os.path.exists(source_file) or force_new:
            self.logger.debug("No catalog found, building from sextractor")
            # Build catalog of point sources
            sextractor = shutil.which('sextractor')
            if sextractor is None:
                sextractor = shutil.which('sex')
                if sextractor is None:
                    raise error.InvalidSystemCommand('sextractor not found')

            if sextractor_params is None:
                sextractor_params = [
                    '-c', '{}/PIAA/resources/conf_files/sextractor/panoptes.sex'.format(os.getenv('PANDIR')),
                    '-CATALOG_NAME', source_file,
                ]

            self.logger.debug("Running sextractor...")
            cmd = [sextractor, *sextractor_params, self.files[image_num]]
            self.logger.debug(cmd)
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

        # Filter point sources near edge
        # w, h = data[0].shape
        w, h = (3476, 5208)

        stamp_size = 50

        top = point_sources['Y'] > stamp_size
        bottom = point_sources['Y'] < w - stamp_size
        left = point_sources['X'] > stamp_size
        right = point_sources['X'] < h - stamp_size

        self._point_sources = point_sources[top & bottom & right & left].to_pandas()

        return self._point_sources

    # def build_all_psc(self):
    #     # Make a data cube for the entire observation
    #     cube = list()

    #     for i, f in enumerate(self.files):
    #         with fits.open(f) as hdu:
    #             d0 = hdu[0].data

    #             stamps = [utils.make_postage_stamp(d0, ps['X'], ps['Y'], padding=self.stamp_padding).flatten()
    #                       for ps in self.point_sources]

    #             cube.append(stamps)

    #             hdu.close()

    #     self.psc_collection = np.array(cube)

    # def get_variance(self, frame, i, j):
    #     """ Compare one stamp to another and get variance

    #     Args:
    #         stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
    #         frame(int): The frame number we want to compare
    #         i(int): Index of target PIC
    #         j(int): Index of PIC we want to compare target to
    #     """

    #     normal_target = self.psc_collection[frame, i] / self.psc_collection[frame, i].sum()
    #     normal_compare = self.psc_collection[frame, j] / self.psc_collection[frame, j].sum()

    #     normal_diff = (normal_target - normal_compare)**2

    #     diff_sum = normal_diff.sum()

    #     return diff_sum

    # def get_all_variance(self, i):
    #     """ Get all variances for given target

    #     Args:
    #         stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
    #         i(int): Index of target PIC
    #     """

    #     v = np.zeros((self.num_stars))

    #     for m in range(self.num_frames):
    #         for j in range(self.num_stars):
    #             v[j] += self.get_variance(m, i, j)

    #     s = pd.Series(v)
    #     return s

    def _get_stamp_points(self, idx):
        # Print beginning, middle, and end positions
        start_pos = self.pixel_locations.iloc[0, idx]
        mid_pos = self.pixel_locations.iloc[int(len(self.files) / 2), idx]
        end_pos = self.pixel_locations.iloc[-1, idx]

        return start_pos, mid_pos, end_pos

    def _adjust_stamp_midpoint(self, mid_pos):
        """ The midpoint pixel should always end up as Blue to accommodate slicing """
        color = utils.pixel_color(mid_pos[0], mid_pos[1])

        x = mid_pos[0]
        y = mid_pos[1]

        if color == 'G2':
            x -= 1
        elif color == 'G1':
            y -= 1
        elif color == 'B':
            pass
        elif color == 'R':
            x += 1
            y += 1

        return (x, y)

    def _nearest_10(self, num):
        """ Get the nearest 10 block """
        return int(np.ceil(np.abs(num) / 10)) * 10

    def _load_images(self, remove_pointing=True):
        seq_files = glob("{}/*.fits".format(self.image_dir))
        seq_files.sort()

        self.files = seq_files
