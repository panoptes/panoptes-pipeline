import os

from contextlib import suppress

import numpy as np
import pandas as pd

from scipy import linalg
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter

from tqdm import tqdm

import csv
from dateutil.parser import parse as parse_date
from astropy.io import fits
from astropy.stats import SigmaClip

from photutils import Background2D
from photutils import MeanBackground
from photutils import MMMBackground
from photutils import MedianBackground
from photutils import SExtractorBackground
from photutils import BkgZoomInterpolator

from panoptes.utils.images import fits as fits_utils
from panoptes.utils import bayer

from panoptes.piaa.utils import helpers
from panoptes.piaa.utils import plot

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def normalize(cube):
    return (cube.T / cube.sum(1)).T


def get_psc(picid, stamps, frame_slice=None):
    try:
        psc = np.array(stamps[picid]['data'])
    except KeyError:
        raise Exception("{} not found in the stamp collection.".format(picid))

    if frame_slice is not None:
        psc = psc[frame_slice]

    return psc


def find_similar_stars(
        picid,
        stamps,
        csv_file=None,
        camera_bias=2048,
        num_refs=100,
        snr_limit=10,
        show_progress=True,
        force_new=False,
        *args, **kwargs):
    """ Get all variances for given target

    Args:
        stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
        i(int): Index of target PIC
    """
    logger.info("Finding similar stars for PICID {}".format(picid))

    if force_new and csv_file and os.path.exist(csv_file):
        logger.info("Forcing new file for {}".format(picid))
        with suppress(FileNotFoundError):
            os.remove(csv_file)

    try:
        df0 = pd.read_csv(csv_file, index_col=[0])
        logger.info("Found existing csv file: {}".format(df0))
        return df0
    except Exception:
        pass

    data = dict()

    logger.info("Getting Target PSC and subtracting bias")
    psc0 = get_psc(picid, stamps, **kwargs) - camera_bias
    logger.info("Target PSC shape: {}".format(psc0.shape))
    num_frames = psc0.shape[0]

    # Normalize
    logger.info("Normalizing target for {} frames".format(num_frames))
    normalized_psc0 = np.zeros_like(psc0, dtype='f4')

    good_frames = []
    for frame_index in range(num_frames):
        try:
            if psc0[frame_index].sum() > 0.:
                # Normalize and store frame
                normalized_psc0[frame_index] = psc0[frame_index] / psc0[frame_index].sum()

                # Save frame index
                good_frames.append(frame_index)
            else:
                logger.warning("Sum for target frame {} is 0".format(frame_index))
        except RuntimeWarning:
            logger.warning("Skipping frame {}".format(frame_index))

    iterator = enumerate(list(stamps.keys()))
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(stamps),
            desc="Finding similar",
            leave=False
        )

    for i, source_index in iterator:
        # Skip low SNR (if we know SNR)
        try:
            snr = float(stamps[source_index].attrs['snr'])
            if snr < snr_limit:
                logger.info("Skipping PICID {}, low snr {:.02f}".format(source_index, snr))
                continue
        except KeyError:
            logger.debug("No source in table: {}".format(picid))
            pass

        try:
            psc1 = get_psc(source_index, stamps, **kwargs) - camera_bias
        except Exception:
            continue

        normalized_psc1 = np.zeros_like(psc1, dtype='f4')

        # Normalize
        for frame_index in good_frames:
            if psc1[frame_index].sum() > 0.:
                normalized_psc1[frame_index] = psc1[frame_index] / psc1[frame_index].sum()

        # Store in the grid
        try:
            v = ((normalized_psc0 - normalized_psc1) ** 2).sum()
            data[source_index] = v
        except ValueError as e:
            logger.info("Skipping invalid stamp for source {}: {}".format(source_index, e))

    df0 = pd.DataFrame(
        {'v': list(data.values())},
        index=list(data.keys())).sort_values(by='v')

    if csv_file:
        df0[:num_refs].to_csv(csv_file)

    return df0


def get_ideal_full_coeffs(stamp_collection):

    num_frames = stamp_collection.shape[1]
    num_pixels = stamp_collection.shape[2]

    target_frames = stamp_collection[0].flatten()
    refs_frames = stamp_collection[1:].reshape(-1, num_frames * num_pixels).T

    coeffs = linalg.lstsq(refs_frames, target_frames)

    return coeffs


def get_ideal_full_psc(stamp_collection, coeffs):
    refs = stamp_collection[1:]
    created_frame = (refs.T * coeffs).sum(2).T
    return created_frame


def get_aperture_sums(psc0,
                      psc1,
                      image_times,
                      readout_noise=10.5,
                      separate_green=False,
                      plot_apertures=False,
                      aperture_plot_path=None,
                      aperture_cutoff=2e-1,
                      picid=None,
                      ):
    """Perform differential aperture photometry on the given PSCs.

    `psc0` and `psc1` are Postage Stamp Cubes (PSC) of N frames x M
    pixels, where M = width x height of the stamp and is assumed to be
    square.

    For each N frame, an aperture is placed around the source in `psc0`
    and the corresponding pixel location in `psc1`. This aperture cutout
    is then split on color channels and for each channel the sum of
    the target, the sum of the reference, and the difference is given.

    ..todo::

        Adaptive Aperture:

    Args:
        psc0 (`numpy.array`): An NxM cube of source postage stamps.
        psc1 (`numpy.array`): An NxM cube to be used as the comparison.
        image_times (list(`datetime`)): A list of `datetime.datetime` objects to
            be used for an index.
        readout_noise (float): Readout noise in e- / pixel, default 10.5.
        separate_green (bool): If separate green color channels should be created,
            default False. If True, the G2 pixel is marked as `c`.
        plot_apertures (bool, optional): If a figure should be generated showing
            each of the aperture stamps, default False.

    Returns:
        `pandas.DataFrame`: A dataframe with `color`, `target`, and `reference`.
            columns.

    """
    num_frames, stamp_size = psc0.shape

    stamp_side = int(np.sqrt(stamp_size))

    apertures = list()
    diff = list()
    for frame_idx, image_time in zip(range(num_frames), image_times):

        # Get target and reference stamp for this frame
        t0 = psc0[frame_idx].reshape(stamp_side, stamp_side)
        i0 = psc1[frame_idx].reshape(stamp_side, stamp_side)

        logger.debug(f'Using adaptive apertures')
        pixel_locations = helpers.get_adaptive_aperture(i0, cutoff_value=aperture_cutoff)
        logger.debug(f'Frame {frame_idx} pixel locations: {pixel_locations}')

        for color, pixel_loc in pixel_locations.items():
            target_pixel_values = np.array([t0[loc[0], loc[1]] for loc in pixel_loc])
            ideal_pixel_values = np.array([i0[loc[0], loc[1]] for loc in pixel_loc])

            # Get the sum in electrons
            t_sum = target_pixel_values.sum()
            i_sum = ideal_pixel_values.sum()

            # Add the noise
            target_photon_noise = np.sqrt(t_sum)
            ideal_photon_noise = np.sqrt(i_sum)

            readout = readout_noise * len(pixel_loc)

            # TODO Scintillation noise?

            target_total_noise = np.sqrt(target_photon_noise**2 + readout**2)
            ideal_total_noise = np.sqrt(ideal_photon_noise**2 + readout**2)

            # Record the values.
            diff.append({
                'color': color,
                'target': t_sum,
                'target_err': target_total_noise,
                'reference': i_sum,
                'reference_err': ideal_total_noise,
                'obstime': image_time,
                'aperture_pixels': pixel_loc,
            })

    # Light-curve dataframe
    lc0 = pd.DataFrame(diff).set_index(['obstime'])

    if plot_apertures:
        os.makedirs(aperture_plot_path, exist_ok=True)
        plot.make_apertures_plot(apertures, output_dir=aperture_plot_path)

    return lc0


def get_imag(x, t=1):
    """Instrumental magnitude.

    Args:
        x (float|list(float)): Flux values.
        t (int, optional): Exposure time.

    Returns:
        float|list(float): Instrumental magnitudes.
    """
    return -2.5 * np.log10(x / t)


def normalize_lightcurve(lc0, method='median', use_frames=None):
    """Normalize the lightcurve data, including errors.

    Args:
        lc0 (`pandas.DataFrame`): Dataframe with light curve values.
        method (str, optional): The normalization method used, either `median` (default) or `mean`.
        use_frames (None, optional): A `slice` object to select frame for normalization, e.g. the
            pre-ingress or post-egress frames.

    Returns:
        `pandas.DataFrame`: A copy of the dataframe with normalized light curve values and error.
    """
    # Make a copy
    lc1 = lc0.copy()

    methods = {
        'median': lambda x: np.ma.median(x),
        'mean': lambda x: np.ma.mean(x)
    }

    use_method = methods[method]
    if use_frames is None:
        use_frames = slice(None)

    data_to_normalize = lc1[use_frames].groupby('color')

    for field in ['reference', 'target']:
        field_to_normalize = data_to_normalize[field]

        # Apply the normalization.
        normalization_values = field_to_normalize.apply(use_method)

        for color, norm_value in normalization_values.iteritems():

            # Get the raw values.
            raw_values = lc1.loc[lc1.color == color, (f'{field}')]
            raw_error = lc1.loc[lc1.color == color, (f'{field}_err')]

            # Replace with normalized versions.
            lc1.loc[lc1.color == color, (f'{field}')] = (raw_values / norm_value)
            lc1.loc[lc1.color == color, (f'{field}_err')] = (raw_error / norm_value)

    return lc1


def get_diff_flux(lc0,
                  smooth=False,
                  savgol_polyorder=None,
                  savgol_sigma=3,
                  sigma_cutoff=None
                  ):

    lc1 = lc0.copy()
    lc1['flux'] = np.nan
    lc1['flux_err'] = np.nan

    for i, color in enumerate('rgb'):
        # Get the normalized flux for each channel
        color_data = lc1.loc[lc1.color == color]

        # Target and error
        t0 = color_data.target
        t0_err = color_data.target_err

        # Reference and error
        r0 = color_data.reference
        r0_err = color_data.reference_err

        # Get the differential flux and error
        flux = t0 / r0
        flux_err = np.sqrt((t0_err / t0)**2 + (r0_err / r0)**2)
        flux_index = color_data.index

        if savgol_polyorder:
            if savgol_polyorder % 2 == 0:
                window_size = savgol_polyorder + 1
            else:
                window_size = savgol_polyorder + 2

            filter0 = savgol_filter(flux, window_size, polyorder=savgol_polyorder)
            flux = (flux - filter0)

            # Clip the filtered
            flux = sigma_clip(flux, sigma=savgol_sigma)

            # Add back the filter
            flux = flux + filter0

            flux_err = np.ma.array(flux_err, mask=flux.mask)
            flux_index = np.ma.array(color_data.index, mask=flux.mask)

        if sigma_cutoff:
            # Sigma clip the differential flux
            flux = sigma_clip(flux, sigma=sigma_cutoff)
            flux_err = np.ma.array(flux_err, mask=flux.mask)
            flux_index = np.ma.array(color_data.index, mask=flux.mask, dtype=bool)

        # Basic correction
        if smooth:
            window_size = len(flux)
            if window_size % 2 == 0:
                window_size -= 1
            smooth1 = savgol_filter(flux, window_size, polyorder=1)
            flux = (flux - smooth1) + 1

        lc1.loc[lc1.color == color, ('flux')] = flux.filled(np.nan)
        lc1.loc[lc1.color == color, ('flux_err')] = flux_err.filled(np.nan)

    return lc1, flux.mask


def subtract_color_background(fits_fn,
                              bucket_path=None,
                              box_size=(84, 84),
                              filter_size=(3, 3),
                              camera_bias=2048,
                              estimator='median',
                              interpolator='zoom',
                              sigma=5,
                              iters=5,
                              exclude_percentile=100
                              ):
    """Get the background for each color channel.

    Most of the options are described in the `photutils.Background2D` page:

    https://photutils.readthedocs.io/en/stable/background.html#d-background-and-noise-estimation

    Args:
        fits_fn (str): The filename of the FITS image.
        bucket_path (None, optional): Bucket path for upload, no upload if None (default).
        box_size (tuple, optional): The box size over which to compute the
            2D-Background, default (84, 84).
        filter_size (tuple, optional): The filter size for determining the median,
            default (3, 3).
        camera_bias (int, optional): The built-in camera bias, default 2048.
        estimator (str, optional): The estimator object to use, default 'median'.
        interpolator (str, optional): The interpolater object to user, default 'zoom'.
        sigma (int, optional): The sigma on which to filter values, default 5.
        iters (int, optional): The number of iterations to sigma filter, default 5.
        exclude_percentile (int, optional): The percentage of the data (per channel)
            that can be masked, default 100 (i.e. all).

    Returns:
        list: A list containing a `photutils.Background2D` for each color channel, in RGB order.
    """
    estimators = {
        'sexb': SExtractorBackground,
        'median': MedianBackground,
        'mean': MeanBackground,
        'mmm': MMMBackground
    }
    interpolators = {
        'zoom': BkgZoomInterpolator(),
    }

    print(f"Performing background subtraction for {fits_fn}")
    print(f"Est: {estimator} Interp: {interpolator} Box: {box_size} Sigma: {sigma} Iters: {iters}")

    data = fits.getdata(fits_fn) - camera_bias
    header = fits_utils.getheader(fits_fn)

    # Got the data per color channel.
    rgb_data = get_color_data(data)

    bkg_estimator = estimators[estimator]()
    interp = interpolators[interpolator]
    sigma_clip = SigmaClip(sigma=sigma, maxiters=iters)

    backgrounds = list()
    for color, color_data in zip(['R', 'G', 'B'], rgb_data):
        print(f'Performing background {color} for {fits_fn}')

        bkg = Background2D(color_data, box_size, filter_size=filter_size,
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                           exclude_percentile=exclude_percentile,
                           mask=color_data.mask,
                           interpolator=interp)

        # Create a masked array for the background
        backgrounds.append(np.ma.array(data=bkg.background, mask=color_data.mask))
        print(f"{color} Value: {bkg.background_median:.02f} RMS: {bkg.background_rms_median:.02f}")

    # Create one array for the backgrounds, where any holes are filled with zeros.
    # Add bias back to data so we can save with unsigned.
    full_background = np.ma.array(backgrounds).sum(0).filled(0)

    # Upload background file
    if bucket_path is not None:
        try:
            back_fn = fits_fn.replace('.fits', '-background.fits')
            bucket_path = bucket_path.replace('.fits', '-background.fits')

            # Make FITS file with background substracted version
            save_back_data = (full_background + camera_bias).astype(np.uint16)
            hdu = fits.PrimaryHDU(data=save_back_data, header=header)
            hdu.writeto(back_fn, overwrite=True)

            # Pack the background
            back_fz_fn = fits_utils.fpack(back_fn)
            print(f'Background file saved to {back_fz_fn}')

        except Exception as e:
            print(f'Error uploading background file for {fits_fn}: {e}')

    # Subtract the background
    subtacted_data = data - full_background

    # Replace FITS file with subtracted version
    try:
        header['BKGSUB'] = True
        hdu = fits.PrimaryHDU(data=subtacted_data.astype(np.int16), header=header)
        hdu.writeto(fits_fn, overwrite=True)
    except Exception as e:
        print(f'Error writing substracted FITS: {e}')

    return fits_fn


def get_color_data(data):
    """Split the data according to the RGB Bayer pattern.

    Args:
        data (`numpy.array`): The image data.

    Returns:
        list: A list contained an `numpy.ma.array` for each color channel.
    """
    red_pixels_mask = np.ones_like(data)
    green_pixels_mask = np.ones_like(data)
    blue_pixels_mask = np.ones_like(data)

    red_pixels_mask[1::2, 0::2] = False  # Red
    green_pixels_mask[1::2, 1::2] = False  # Green
    green_pixels_mask[0::2, 0::2] = False  # Green
    blue_pixels_mask[0::2, 1::2] = False  # Blue

    red_data = np.ma.array(data, mask=red_pixels_mask)
    green_data = np.ma.array(data, mask=green_pixels_mask)
    blue_data = np.ma.array(data, mask=blue_pixels_mask)

    rgb_data = [
        red_data,
        green_data,
        blue_data
    ]

    return rgb_data


def get_postage_stamps(point_sources, fits_fn, stamp_size=10, tmp_dir='/tmp'):
    """Extract postage stamps for each PICID in the given file.

    Args:
        point_sources (`pandas.DataFrame`): A DataFrame containing the results from `sextractor`.
        fits_fn (str): The name of the FITS file to extract stamps from.
        stamp_size (int, optional): The size of the stamp to extract, default 10 pixels.
    """
    data = fits.getdata(fits_fn)
    header = fits.getheader(fits_fn)

    print(f'Extracting {len(point_sources)} point sources from {fits_fn}')

    row = point_sources.iloc[0]
    sources_csv_fn = os.path.join(
        tmp_dir, f'{row.unit_id}-{row.camera_id}-{row.seq_time}-{row.img_time}.csv')
    print(f'Sources metadata will be extracted to {sources_csv_fn}')

    print(f'Starting source extraction for {fits_fn}')
    with open(sources_csv_fn, 'w') as metadata_fn:
        writer = csv.writer(metadata_fn, quoting=csv.QUOTE_MINIMAL)

        # Write out headers.
        csv_headers = [
            'picid',
            'unit_id',
            'camera_id',
            'sequence_time',
            'image_time',
            'x', 'y',
            'ra', 'dec',
            'tmag', 'tmag_err',
            'vmag', 'vmag_err',
            'lumclass', 'lum', 'lum_err',
            'contratio', 'numcont',
            'catalog_sep_arcsec',
            'sextractor_flags',
            # 'sextractor_background',
            'slice_y',
            'slice_x',
            'exptime',
            'field',
            'bucket_path',
        ]
        csv_headers.extend([f'pixel_{i:02d}' for i in range(stamp_size**2)])
        writer.writerow(csv_headers)

        for picid, row in point_sources.iterrows():
            # Get the stamp for the target
            target_slice = bayer.get_stamp_slice(
                row.x, row.y,
                stamp_size=(stamp_size, stamp_size),
                ignore_superpixel=False,
                verbose=False
            )

            # Add the target slice to metadata to preserve original location.
            row['target_slice'] = target_slice
            stamp = data[target_slice].flatten().tolist()

            row_values = [
                int(picid),
                str(row.unit_id),
                str(row.camera_id),
                parse_date(row.seq_time),
                parse_date(row.img_time),
                int(row.x), int(row.y),
                row.ra, row.dec,
                row.tmag, row.tmag_err,
                row.vmag, row.vmag_err,
                row.lumclass, row.lum, row.lum_err,
                row.contratio, row.numcont,
                row.catalog_sep_arcsec,
                int(row['flags']),
                # row.background,
                target_slice[0],
                target_slice[1],
                header.get('EXPTIME', -1),
                header.get('FIELD', 'UNKNOWN'),
                row.bucket_path,
                *stamp
            ]

            # Write out stamp data
            writer.writerow(row_values)

    print(f'PSC file saved to {sources_csv_fn}')
    return sources_csv_fn
