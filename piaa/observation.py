import logging
import os
import shutil
import subprocess

from glob import glob

from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS

from photutils import RectangularAperture
from photutils import aperture_photometry
from photutils import make_source_mask

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from . import utils
from pocs.utils import error


class Observation(object):

    def __init__(self, image_dir, aperture_size=6, camera_bias=1024):
        """ A sequence of images to be processed as one observation """
        assert os.path.exists(image_dir), "Specified directory does not exist"

        logging.basicConfig(filename='/var/panoptes/logs/piaa.log', level=logging.DEBUG)
        self.logger = logging
        self.logger.info('Setting up Observation for analysis')

        super(Observation, self).__init__()
        self._image_dir = image_dir

        self.camera_bias = camera_bias

        self._img_w = 3476
        self._img_h = 5208

        self.aperture_size = aperture_size

        self._point_sources = None
        self._pixel_locations = None

        self._cutout_masks = (None, None, None)
        self._cutouts_cache = {}

        # self.psc_collection = None

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

    # @property
    # def num_frames(self):
    #     assert self.psc_collection is not None
    #     return self.psc_collection.shape[0]

    # @property
    # def num_stars(self):
    #     assert self.psc_collection is not None
    #     return self.psc_collection.shape[1]

    def subtract_background(self, cutout, r_mask=None, g_mask=None, b_mask=None, background_sub_method='median'):
        """ Perform RGB background subtraction

        Args:
            cutout (numpy.array): A cutout of the data
            r_mask (numpy.ma.array, optional): A mask of the R channel
            g_mask (numpy.ma.array, optional): A mask of the G channel
            b_mask (numpy.ma.array, optional): A mask of the B channel
            background_sub_method (str, optional): Subtraction method of `median` or `mean`

        Returns:
            numpy.array: The background subtracted data recomined into one array
        """
        self.logger.debug("Subtracting background - {}".format(background_sub_method))
        source_mask = make_source_mask(cutout.data, snr=3., npixels=2)

        if r_mask is None or g_mask is None or b_mask is None:
            self.logger.debug("Making RGB masks for data subtraction")
            self._cutout_masks = utils.make_masks(cutout.data)
            r_mask, g_mask, b_mask = self._cutout_masks

        method_lookup = {
            'mean': 0,
            'median': 1,
        }
        method_idx = method_lookup[background_sub_method]

        r_masked_data = np.ma.array(cutout.data, mask=np.logical_or(source_mask, ~r_mask))
        r_stats = sigma_clipped_stats(r_masked_data, sigma=3.)
        r_masked_data = np.ma.array(cutout.data, mask=~r_mask) - int(r_stats[method_idx])

        g_masked_data = np.ma.array(cutout.data, mask=np.logical_or(source_mask, ~g_mask))
        g_stats = sigma_clipped_stats(g_masked_data, sigma=3.)
        g_masked_data = np.ma.array(cutout.data, mask=~g_mask) - int(g_stats[method_idx])

        b_masked_data = np.ma.array(cutout.data, mask=np.logical_or(source_mask, ~b_mask))
        b_stats = sigma_clipped_stats(b_masked_data, sigma=3.)
        b_masked_data = np.ma.array(cutout.data, mask=~b_mask) - int(b_stats[method_idx])

        subtracted_data = r_masked_data.filled(0) + g_masked_data.filled(0) + b_masked_data.filled(0)
        subtracted_data[subtracted_data > 1e3] = 1e-5

        return subtracted_data

    def get_cutouts(self, source_index, force_new=False, *args, **kwargs):
        """ Create a cutout (stamp) of the data

        This uses the start and end points from the source drift to figure out
        an appropriate size to cutout. Data is bias and background subtracted.
        """
        try:
            if force_new:
                del self._cutouts_cache[source_index]
            cutouts = self._cutouts_cache[source_index]
        except KeyError:
            cutouts = list()

            start_pos, mid_pos, end_pos = self._get_cutout_points(source_index)
            mid_pos = self._adjust_cutout_midpoint(mid_pos)

            for i in range(len(self.files)):
                # Get all the data and subtract the bias
                data = fits.getdata(self.files[i]) - self.camera_bias

                # Get the width and height of data region
                width, height = (start_pos - end_pos)

                # Values should be padded at x4
                cutout = Cutout2D(
                    data, (mid_pos[0], mid_pos[1]), (self._nearest_10(height) + 8, self._nearest_10(width) + 4))

                if i == 0:
                    r_mask, g_mask, b_mask = utils.make_masks(cutout.data)

                cutout.data = self.subtract_background(cutout, r_mask, g_mask, b_mask, *args, **kwargs)
                cutouts.append(cutout)

            self._cutouts_cache[source_index] = cutouts

        return cutouts

    def get_fluxes(self, source_index):

        fluxes = []

        cutouts = self.get_cutouts(source_index)

        # Get aperture photometry
        for i in self.pixel_locations:
            x = int(self.pixel_locations[i, source_index, 0] - cutouts[i].origin_original[0]) - 0.5
            y = int(self.pixel_locations[i, source_index, 1] - cutouts[i].origin_original[1]) - 0.5

            aperture = RectangularAperture((x, y), w=6, h=6, theta=0)

            phot_table = aperture_photometry(cutouts[i].data, aperture)

            flux = phot_table['aperture_sum'][0]

            fluxes.append(flux)

        fluxes = np.array(fluxes)

        return fluxes

    def get_frame_stamp(self, source_index, frame_index, *args, **kwargs):

        cutouts = self.get_cutouts(source_index, *args, **kwargs)

        cutout = cutouts[frame_index]

        return cutout

    def get_aperture(self, source_index, frame_index, *args, **kwargs):
        cutout = self.get_frame_stamp(source_index, frame_index, *args, **kwargs)

        x = int(self.pixel_locations[frame_index, source_index, 0] - cutout.origin_original[0]) - 0.5
        y = int(self.pixel_locations[frame_index, source_index, 1] - cutout.origin_original[1]) - 0.5

        aperture = RectangularAperture((x, y), w=6, h=6, theta=0)

        return aperture

    def plot_stamp(self, source_index, frame_index, show_bayer=True, show_data=False, *args, **kwargs):

        cutout = self.get_frame_stamp(source_index, frame_index, *args, **kwargs)

        fig, (ax1) = plt.subplots(1, 1, facecolor='white')
        fig.set_size_inches(30, 15)

        aperture = self.get_aperture(source_index, frame_index, return_aperture=True)

        aperture_mask = aperture.to_mask(method='center')[0]
        aperture_data = aperture_mask.cutout(cutout.data)

        phot_table = aperture_photometry(cutout.data, aperture, method='center')

        if show_data:
            print(np.flipud(aperture_data))  # Flip the data to match plot

        cax = ax1.imshow(cutout.data, cmap='cubehelix_r', vmin=0., vmax=aperture_data.max())
        fig.colorbar(cax)

        aperture.plot(color='b', ls='--', lw=2)

        if show_bayer:
            # Bayer pattern
            for i, val in np.ndenumerate(cutout.data):
                x, y = cutout.to_original_position((i[1], i[0]))
                ax1.text(x=i[1], y=i[0], ha='center', va='center',
                         s=utils.pixel_color(x, y, zero_based=True), fontsize=10, alpha=0.25)

            # major ticks every 2, minor ticks every 1
            x_major_ticks = np.arange(-0.5, cutout.bbox_cutout[1][1], 2)
            x_minor_ticks = np.arange(-0.5, cutout.bbox_cutout[1][1], 1)

            y_major_ticks = np.arange(-0.5, cutout.bbox_cutout[0][1], 2)
            y_minor_ticks = np.arange(-0.5, cutout.bbox_cutout[0][1], 1)

            ax1.set_xticks(x_major_ticks)
            ax1.set_xticks(x_minor_ticks, minor=True)
            ax1.set_yticks(y_major_ticks)
            ax1.set_yticks(y_minor_ticks, minor=True)

            ax1.grid(which='major', color='r', linestyle='-', alpha=0.25)
            ax1.grid(which='minor', color='r', linestyle='-', alpha=0.1)

        ax1.set_title("Source {} Frame {} Aperture Flux: {}".format(source_index,
                                                                    frame_index, phot_table['aperture_sum'][0]))

        return fig

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

    def _get_cutout_points(self, idx):
        # Print beginning, middle, and end positions
        start_pos = self.pixel_locations.iloc[0, idx]
        mid_pos = self.pixel_locations.iloc[int(len(self.files) / 2), idx]
        end_pos = self.pixel_locations.iloc[-1, idx]

        return start_pos, mid_pos, end_pos

    def _adjust_cutout_midpoint(self, mid_pos):
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
