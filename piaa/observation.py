import os
import shutil
import subprocess

from warnings import warn

from collections import namedtuple
from glob import glob

from astropy.io import fits
from astropy.table import Table
from astropy.utils.console import ProgressBar
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import WCS

from photutils import Background2D
from photutils import MedianBackground
from photutils import RectangularAperture
from photutils import SigmaClip
from photutils import aperture_photometry
from photutils import find_peaks

import h5py
import numpy as np
import pandas as pd

from scipy.optimize import minimize

from matplotlib import gridspec
from matplotlib import patches
from matplotlib import pyplot as plt

from . import utils


PSC = namedtuple('PSC', ['data', 'mask'])


class Observation(object):

    def __init__(self, image_dir, aperture_size=6, camera_bias=2048,
                 log_level='INFO', build_data_cube=True, *args, **kwargs):
        """ A sequence of images to be processed as one observation """
        assert os.path.exists(image_dir), "Specified directory does not exist"

        self.verbose = False
        if kwargs.get('verbose', False):
            self.verbose = True

        if image_dir.endswith('/'):
            image_dir = image_dir[:-1]
        self._image_dir = image_dir

        self.log('*' * 80)
        self.log('Setting up Observation for analysis')

        super(Observation, self).__init__()

        self.camera_bias = camera_bias

        self._img_h = 3476
        self._img_w = 5208

        # Background estimation boxes
        self.background_box_h = 316
        self.background_box_w = 434

        self.background_region = {}

        self.aperture_size = aperture_size
        self.stamp_size = (None, None)

        self._point_sources = None
        self._pixel_locations = None

        self._total_integration_time = None

        self._wcs = None

        self._rgb_masks = None

        self._stamp_masks = (None, None, None)
        self._stamps_cache = {}

        self._hdf5 = None
        self._hdf5_stamps = None
        self.slices = dict()

        self._load_images()

        if build_data_cube:
            assert self.data_cube is not None

    @property
    def hdf5(self):
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self.image_dir + '.hdf5')

        return self._hdf5

    @property
    def hdf5_stamps(self):
        if self._hdf5_stamps is None:
            self._hdf5_stamps = h5py.File(self.image_dir + '_stamps.hdf5')

        return self._hdf5_stamps

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

    @property
    def total_integration_time(self):
        if self._total_integration_time is None:
            self._total_integration_time = np.array(
                [np.round(float(fits.getval(f, 'EXPTIME'))) for f in self.files]).sum()

        return self._total_integration_time

    @property
    def pixel_locations(self):
        if self._pixel_locations is None:
            # Get RA/Dec coordinates from first frame
            ra = self.point_sources['ALPHA_J2000']
            dec = self.point_sources['DELTA_J2000']

            locs = list()

            for f in self.files:
                wcs = WCS(f)
                xy = np.array(wcs.all_world2pix(ra, dec, 0, ra_dec_order=True))

                # Transpose
                locs.append(xy.T)

            locs = np.array(locs)
            self._pixel_locations = pd.Panel(locs)

        return self._pixel_locations

    @property
    def wcs(self):
        if self._wcs is None:
            self._wcs = WCS(self.files[0])

        return self._wcs

    @property
    def num_frames(self):
        return len(self.files)

    @property
    def num_point_sources(self):
        return len(self.point_sources.index)

    @property
    def data_cube(self):
        try:
            cube_dset = self.hdf5['cube']
        except KeyError:
            if self.verbose:
                self.log("Creating data cube", end='')
            cube_dset = self.hdf5.create_dataset('cube', (len(self.files), self._img_h, self._img_w))
            for i, f in enumerate(self.files):
                if self.verbose and i % 10 == 0:
                    self.log('.', end='')

                cube_dset[i] = fits.getdata(f) - self.camera_bias

            self.log('Done')

        return cube_dset

    @property
    def rgb_masks(self):
        """ RGB Mask arrays

        Read the RGB masks from a stored file or generate and save to file accordingly

        Returns:
            numpy.MaskedArray: A 3xNxM array where the first axis corresponds to color
                and the NxM is the full frame size
        """
        if self._rgb_masks is None:
            rgb_mask_file = '{}/rgb_masks.numpy'.format(os.getenv('PANDIR'))
            try:
                self._rgb_masks = np.load(rgb_mask_file)
            except FileNotFoundError:
                self.log("Making RGB masks")
                self._rgb_masks = np.array(utils.make_masks(self.data_cube[0]))
                self._rgb_masks.dump(rgb_mask_file)

        return self._rgb_masks

    def log(self, msg, **kwargs):
        if self.verbose:
            if 'end' in kwargs:
                print(msg, end=kwargs['end'])
            else:
                print(msg)

    def get_header_value(self, frame_index, header):
        return fits.getval(self.files[frame_index], header)

    def subtract_background(self, frames=None, display_progress=False, clip_sigma=3., clip_iters=5, **kwargs):
        """Get background estimates for all frames for each color channel

        The first step is to figure out a box size for the background calculations.
        This should be larger enough to encompass background variations while also
        being an even multiple of the image dimensions. We also want them to be
        multiples of a superpixel (2x2 regular pixel) in each dimension.
        The camera for `PAN001` has image dimensions of 5208 x 3476, so
        in order to get an even multiple in both dimensions we remove 60 pixels
        from the width of the image, leaving us with dimensions of 5148 x 3476,
        allowing us to use a box size of 468 x 316, which will create 11
        boxes in each direction.

        We use a 3 sigma median background clipped estimator.
        The built-in camera bias (2048) has already been removed from the data.

        Args:
            frames (list, optional): List of frames to get estimates for, defaults
                to all frames

        """
        if frames is None:
            frames = range(len(self.files))

        sigma_clip = SigmaClip(sigma=clip_sigma, iters=clip_iters)
        bkg_estimator = MedianBackground()

        if display_progress:
            frames_iter = ProgressBar(frames, ipython_widget=kwargs.get('ipython_widget', False))
        else:
            frames_iter = frames

        for frame_index in frames_iter:
            frame_key = 'frame/{}'.format(frame_index)

            # Figure out if we have already subtracted this frame
            try:
                already_subtracted = bool(self.data_cube.attrs[frame_key])
            except KeyError:
                already_subtracted = False

            if already_subtracted is not True:
                self.log("Getting background estimates for frame: {}".format(frame_index))

                background_data = self.get_frame_background(
                    frame_index, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

                self.data_cube[frame_index] -= background_data

                self.data_cube.attrs[frame_key] = True

    def get_frame_background(self, frame_index, sigma_clip=None,
                             bkg_estimator=None, summary=False, background_obj=False, clip_iters=5, clip_sigma=3.):
        frame_key = 'frame/{}'.format(frame_index)

        # Figure out if we have already subtracted this frame
        try:
            already_subtracted = bool(self.data_cube.attrs[frame_key])
        except KeyError:
            already_subtracted = False

        # Return error if trying to return already subtracted background
        if summary is False and already_subtracted:
            raise Exception("Cannot get already subtracted background")

        # Get backgrounds dataset or create
        try:
            background_dset = self.hdf5['background']
        except KeyError:
            self.log("Creating background dataset")
            # Create dataset that will hold the mmedian and the rms_median for 3 channels for all frames
            background_dset = self.hdf5.create_dataset('background', (len(self.files), 3, 2))

        # Figure out if we have already have a summary
        try:
            have_summary = bool(background_dset.attrs[frame_key])
        except KeyError:
            have_summary = False

        # If we just want the summary have it, return
        if summary and have_summary:
            return background_dset[frame_index]

        # Get clip and estimator objects
        if sigma_clip is None:
            sigma_clip = SigmaClip(sigma=clip_sigma, iters=clip_iters)

        if bkg_estimator is None:
            bkg_estimator = MedianBackground()

        # Get the bias subtracted data for the frame
        data = self.data_cube[frame_index]

        # Create holder for the actual background
        background_data = np.zeros_like(data)
        background_objs = dict()

        for color_index, masks in enumerate(zip(['R', 'G', 'B'], self.rgb_masks)):
            color = masks[0]
            mask = masks[1]
            bkg = Background2D(data, (self.background_box_h, self.background_box_w),
                               sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, mask=~mask)

            background_objs[color] = bkg

            self.log("\t{} Background\t Value: {:.02f}\t RMS: {:.02f}".format(
                color, bkg.background_median, bkg.background_rms_median))

            if summary:
                background_dset[frame_index, color_index] = (bkg.background_median, bkg.background_rms_median)
            elif background_obj is False:
                background_masked_data = np.ma.array(bkg.background, mask=~mask)
                background_data += background_masked_data.filled(0)

        # Mark that we already have summary
        self.hdf5['background'].attrs[frame_key] = True

        if summary:
            return background_dset[frame_index]
        elif background_obj is True:
            return background_objs
        else:
            return background_data

    def get_psc(self, source_index, frame_slice=None):
        try:
            ss = self.slices[source_index]

            if frame_slice is None:
                frame_slice = slice(0, self.num_frames)

            data = self.data_cube[frame_slice, ss[0], ss[1]]
            masks = self.rgb_masks[:, ss[0], ss[1]]

            psc = PSC(data=data, mask=masks)

        except KeyError:
            raise Exception("You must run create_stamps first")

        return psc

    def get_ideal_psc(self, stamp_collection, coeffs, target_psc):
        num_frames = target_psc.data.shape[0]
        stamp_h = target_psc.data.shape[1]
        stamp_w = target_psc.data.shape[2]

        ref_frames = []

        for frame_index in range(num_frames):
            refs_frame = stamp_collection[1:, frame_index]
            ref_frame = (refs_frame.T * coeffs[frame_index]).T.sum(0).reshape(stamp_h, stamp_w)
            ref_frames.append(ref_frame)

        return np.array(ref_frames)

    def create_stamp_slices(self, target_index, padding=3, display_progress=False, *args, **kwargs):
        """Create subtracted stamps for entire data cube

        Creates a slice through the cube corresponding to a stamp and stores the
        subtracted data in the hdf5 table with key `stamp/<index>`, where
        `<index>` is the source index from `point_sources`

        Args:
            remove_cube (bool, optional): Remove the full cube from the hdf5 file after
                processing, defaults to False
            *args (TYPE): Description
            **kwargs (dict): `ipython_widget=True` can be passed to display progress
                within a notebook

        """

        self.log("Starting stamp creation")

        r_min, r_max, c_min, c_max = self.get_stamp_bounds(target_index, padding=padding)

        color = utils.pixel_color(c_min, r_max, zero_based=True)
        if color == 'B':
            r_max += 1
            r_min += 1
        elif color == 'R':
            c_min += 1
            c_max += 1
        elif color == 'G2':
            c_min += 1
            c_max += 1
            r_max += 1
            r_min += 1

        height = r_max - r_min
        width = c_max - c_min

        self.log("Stamp size: {} {}".format(height, width))

        if display_progress:
            iterator = ProgressBar(self.point_sources.index, ipython_widget=kwargs.get('ipython_widget', False))
        else:
            iterator = self.point_sources.index

        # try:
        #     del self.hdf5_stamps['stamps']
        # except Exception:
        #     pass

        # stamp_dset = self.hdf5_stamps.create_dataset(
            # 'stamps', (self.num_point_sources, self.num_frames, height, width))

        self.slices = dict()

        for source_index in iterator:

            try:
                r_min, r_max, c_min, c_max = self.get_stamp_bounds(
                    source_index, height=height, width=width, padding=padding)

                color = utils.pixel_color(c_min, r_max, zero_based=True)
                if color == 'B':
                    r_max += 1
                    r_min += 1
                elif color == 'R':
                    c_min += 1
                    c_max += 1
                elif color == 'G2':
                    c_min += 1
                    c_max += 1
                    r_max += 1
                    r_min += 1

                self.slices[source_index] = [slice(r_min, r_max), slice(c_min, c_max)]

            except Exception as e:
                self.log("Problem creating stamp for {}: {}".format(source_index, e))

    def get_stamp_bounds(self, source_index, height=None, width=None, padding=0, **kwargs):
        pix = self.pixel_locations[:, source_index]

        if width is None:
            col_max = int(pix.iloc[0].max()) + padding
            col_min = int(pix.iloc[0].min()) - padding
        else:
            col_max = int(pix.iloc[0].max()) + padding
            col_min = col_max - width

        if height is None:
            row_max = int(pix.iloc[1].max()) + padding
            row_min = int(pix.iloc[1].min()) - padding
        else:
            row_max = int(pix.iloc[1].max()) + padding
            row_min = row_max - height

        return row_min, row_max, col_min, col_max

    def get_variance_for_target(self, target_index, display_progress=True, force_new=True, *args, **kwargs):
        """ Get all variances for given target

        Args:
            stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
            i(int): Index of target PIC
        """
        num_sources = self.num_point_sources

        # Assume no match
        data = np.ones((num_sources, num_sources)) * 99.

        try:
            if force_new:
                del self.hdf5_stamps['vgrid']

            vgrid_dset = self.hdf5_stamps['vgrid']
        except KeyError:
            vgrid_dset = self.hdf5_stamps.create_dataset('vgrid', data=data)

        psc0 = self.get_psc(target_index, frame_slice=kwargs.get('frame_slice', None)).data
        num_frames = psc0.shape[0]

        # Normalize
        self.log("Normalizing target for {} frames".format(num_frames))
        frames = []
        normalized_psc0 = np.zeros_like(psc0)
        for frame_index in range(num_frames):
            try:
                if psc0[frame_index].sum() > 0.:
                    normalized_psc0[frame_index] = psc0[frame_index] / psc0[frame_index].sum()
                    frames.append(frame_index)
            except RuntimeWarning:
                warn("Skipping frame {}".format(frame_index))

        if display_progress:
            iterator = ProgressBar(range(num_sources), ipython_widget=kwargs.get('ipython_widget', False))
        else:
            iterator = range(num_sources)

        for source_index in iterator:
            psc1 = self.get_psc(source_index, frame_slice=kwargs.get('frame_slice', None)).data

            normalized_psc1 = np.zeros_like(psc1)

            # Normalize
            for frame_index in frames:
                normalized_psc1[frame_index] = psc1[frame_index] / psc1[frame_index].sum()

            # Store in the grid
            try:
                vgrid_dset[target_index, source_index] = ((normalized_psc0 - normalized_psc1) ** 2).sum()
            except ValueError:
                self.log("Skipping invalid stamp for source {}".format(source_index))

    def get_stamp_collection(self, target_index, num_refs=25):
        vary = self.hdf5_stamps['vgrid']

        vary_series = pd.Series(vary[target_index])
        vary_series.sort_values(inplace=True)

        stamp_h = self.hdf5_stamps.attrs['stamp_rows']
        stamp_w = self.hdf5_stamps.attrs['stamp_cols']

        stamp_collection = np.array([self.get_psc(idx).data for idx in vary_series.index[0:num_refs]]).reshape(
            num_refs, self.num_frames, stamp_h * stamp_w)

        return stamp_collection

    def get_ideal_coeffs(self, stamp_collection, func=None, display_progress=False, **kwargs):
        coeffs = []

        def minimize_func(refs_coeffs, refs_all_but_frame, target_all_but_frame):
            measured = (refs_coeffs * refs_all_but_frame.T).T.sum(0)
            model = target_all_but_frame

            res = ((model - measured)**2).sum()

            return res

        num_frames = stamp_collection.shape[1]

        if display_progress:
            iterator = ProgressBar(range(num_frames), ipython_widget=kwargs.get('ipython_widget', False))
        else:
            iterator = range(num_frames)

        if func is None:
            func = minimize_func

        for frame_index in iterator:

            target_frame = stamp_collection[0, frame_index]
            target_all_but_frame = np.delete(stamp_collection[0], frame_index, 0)

            refs_frame = stamp_collection[1:, frame_index]
            refs_all_but_frame = np.delete(stamp_collection[1:], frame_index, 1)

            try:
                refs_coeffs = coeffs[-1]
            except IndexError:
                refs_coeffs = np.ones(len(refs_all_but_frame))

            if self.verbose and frame_index == 0:
                print("Target frame shape: {}".format(target_frame.shape))
                print("Target other shape: {}".format(target_all_but_frame.shape))
                print("Refs shape: {}".format(refs_frame.shape))
                print("Refs other shape: {}".format(refs_all_but_frame.shape))
                print("Source coeffs shape: {}".format(refs_coeffs.shape))

            res = minimize(func, refs_coeffs, args=(refs_all_but_frame, target_all_but_frame))

            coeffs.append(res.x)

        return np.array(coeffs)

    def get_stamp_mask(self, source_index, **kwargs):
        r_min, r_max, c_min, c_max = self.get_stamp_bounds(source_index, **kwargs)

        masks = []
        for mask in self.rgb_masks:
            masks.append(mask[r_min:r_max, c_min:c_max])

        return masks

    def get_relative_lightcurve(self, target_psc, refpsf_psc, aperture_size=6):

        depths = {}

        num_frames = target_psc.data.shape[0]

        stamp_h = target_psc.data.shape[1]
        stamp_w = target_psc.data.shape[2]

        for frame_index in range(num_frames):
            target_frame = target_psc.data[frame_index].reshape(stamp_h, stamp_w)
            ref_frame = refpsf_psc[frame_index].reshape(stamp_h, stamp_w)

            t0_peaks = find_peaks(target_frame, np.mean(target_frame) * 5)
            t0_peaks.sort(keys=['peak_value'])
            t0_peak = (t0_peaks['x_peak'][-1], t0_peaks['y_peak'][-1])

            # r0_peaks = find_peaks(ref_frame, np.mean(ref_frame) * 5)
            # r0_peaks.sort(keys=['peak_value'])

            aperture = RectangularAperture(t0_peak, aperture_size, aperture_size, 0)

            for color, mask in zip(['R', 'G', 'B'], target_psc.mask):
                t0_flux = aperture.do_photometry(target_frame, method='subpixel', mask=~mask)[0][0]
                r0_flux = aperture.do_photometry(ref_frame, method='subpixel', mask=~mask)[0][0]

                a0 = t0_flux / r0_flux

                try:
                    depths[color].append(a0)
                except KeyError:
                    depths[color] = [a0]

        return depths

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
        source_file = '{}/point_sources_{:02d}.cat'.format(self.image_dir, image_num)
        self.log("Point source catalog: {}".format(source_file))

        if not os.path.exists(source_file) or force_new:
            self.log("No catalog found, building from sextractor")
            # Build catalog of point sources
            sextractor = shutil.which('sextractor')
            if sextractor is None:
                sextractor = shutil.which('sex')
                if sextractor is None:
                    raise Exception('sextractor not found')

            if sextractor_params is None:
                sextractor_params = [
                    '-c', '{}/PIAA/resources/conf_files/sextractor/panoptes.sex'.format(os.getenv('PANDIR')),
                    '-CATALOG_NAME', source_file,
                ]

            self.log("Running sextractor...")
            cmd = [sextractor, *sextractor_params, self.files[image_num]]
            self.log(cmd)
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

        return self._point_sources

    def plot_stamp(self, source_index, frame_index, show_data=False, *args, **kwargs):

        norm = ImageNormalize(stretch=SqrtStretch())

        stamp_slice = self.get_source_slice(source_index, *args, **kwargs)
        stamp = self.get_frame_stamp(source_index, frame_index, reshape=True, *args, **kwargs)

        fig = plt.figure(1)
        fig.set_size_inches(13, 15)
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])
        ax1 = plt.subplot(gs[:, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 1])
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)
        fig.add_subplot(ax3)

        aperture = self.get_frame_aperture(source_index, frame_index, return_aperture=True)

        aperture_mask = aperture.to_mask(method='center')[0]
        aperture_data = aperture_mask.cutout(stamp)

        phot_table = aperture_photometry(stamp, aperture, method='center')

        if show_data:
            print(np.flipud(aperture_data))  # Flip the data to match plot

        cax1 = ax1.imshow(stamp, cmap='cubehelix_r', norm=norm)
        plt.colorbar(cax1, ax=ax1)

        aperture.plot(color='b', ls='--', lw=2, ax=ax1)

        # Bayer pattern
        for i, val in np.ndenumerate(stamp):
            x, y = stamp_slice.cutout.to_original_position((i[1], i[0]))
            ax1.text(x=i[1], y=i[0], ha='center', va='center',
                     s=utils.pixel_color(x, y, zero_based=True), fontsize=10, alpha=0.25)

        # major ticks every 2, minor ticks every 1
        x_major_ticks = np.arange(-0.5, stamp_slice.cutout.bbox_cutout[1][1], 2)
        x_minor_ticks = np.arange(-0.5, stamp_slice.cutout.bbox_cutout[1][1], 1)

        y_major_ticks = np.arange(-0.5, stamp_slice.cutout.bbox_cutout[0][1], 2)
        y_minor_ticks = np.arange(-0.5, stamp_slice.cutout.bbox_cutout[0][1], 1)

        ax1.set_xticks(x_major_ticks)
        ax1.set_xticks(x_minor_ticks, minor=True)
        ax1.set_yticks(y_major_ticks)
        ax1.set_yticks(y_minor_ticks, minor=True)

        ax1.grid(which='major', color='r', linestyle='-', alpha=0.25)
        ax1.grid(which='minor', color='r', linestyle='-', alpha=0.1)

        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_title("Full Stamp", fontsize=16)

        # RGB values plot

        # Show numbers
        for i, val in np.ndenumerate(aperture_data):
            #     print(i[0] / 10, i[1] / 10, val)
            x_loc = (i[1] / 10) + 0.05
            y_loc = (i[0] / 10) + 0.05

            ax2.text(x=x_loc, y=y_loc,
                     ha='center', va='center', s=int(val), fontsize=12, alpha=0.75, transform=ax2.transAxes)

        ax2.set_xticks(x_major_ticks)
        ax2.set_xticks(x_minor_ticks, minor=True)
        ax2.set_yticks(y_major_ticks)
        ax2.set_yticks(y_minor_ticks, minor=True)

        ax2.grid(which='major', color='r', linestyle='-', alpha=0.25)
        ax2.grid(which='minor', color='r', linestyle='-', alpha=0.1)

        ax2.add_patch(patches.Rectangle(
            (1.5, 1.5),
            6, 6,
            fill=False,
            lw=2,
            ls='dashed',
            edgecolor='blue',
        ))

        r_a_mask, g_a_mask, b_a_mask = utils.make_masks(aperture_data)

        ax2.set_xlim(-0.5, 9.5)
        ax2.set_ylim(-0.5, 9.5)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.imshow(np.ma.array(np.ones((10, 10)), mask=~r_a_mask), cmap='Reds', vmin=0, vmax=4., )
        ax2.imshow(np.ma.array(np.ones((10, 10)), mask=~g_a_mask), cmap='Greens', vmin=0, vmax=4., )
        ax2.imshow(np.ma.array(np.ones((10, 10)), mask=~b_a_mask), cmap='Blues', vmin=0, vmax=4., )
        ax2.set_title("Values", fontsize=16)

        # Contour Plot of aperture

        ax3.contourf(aperture_data, cmap='cubehelix_r', vmin=stamp.min(), vmax=stamp.max())
        ax3.add_patch(patches.Rectangle(
            (1.5, 1.5),
            6, 6,
            fill=False,
            lw=2,
            ls='dashed',
            edgecolor='blue',
        ))
        ax3.add_patch(patches.Rectangle(
            (0, 0),
            9, 9,
            fill=False,
            lw=1,
            ls='solid',
            edgecolor='black',
        ))
        ax3.set_xlim(-0.5, 9.5)
        ax3.set_ylim(-0.5, 9.5)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.grid(False)
        ax3.set_facecolor('white')
        ax3.set_title("Contour", fontsize=16)

        fig.suptitle("Source {} Frame {} Aperture Flux: {}".format(source_index,
                                                                   frame_index, int(phot_table['aperture_sum'][0])),
                     fontsize=20)

        fig.tight_layout(rect=[0., 0., 1., 0.95])
        return fig

    def _load_images(self):
        seq_files = glob("{}/*T*.fits".format(self.image_dir))
        seq_files.sort()

        self.files = seq_files
