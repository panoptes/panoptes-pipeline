import os

from contextlib import suppress

import numpy as np
import pandas as pd

from scipy import linalg
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter

from tqdm import tqdm

from panoptes.utils.logger import get_root_logger
from panoptes.piaa.utils import helpers
from panoptes.piaa.utils import plot

import logging
logger = get_root_logger()
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
        except KeyError as e:
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
