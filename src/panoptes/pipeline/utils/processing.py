import os
import csv
from contextlib import suppress

import numpy as np
import pandas as pd
from scipy import linalg

from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from panoptes.utils.images import bayer
from panoptes.utils.logging import logger

from tqdm import tqdm


def normalize(cube):
    return (cube.T / cube.sum(1)).T


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
        (int): Index of target PIC
    """
    logger.info(f"Finding similar stars for PICID {picid}")

    if force_new and csv_file and os.path.exists(csv_file):
        logger.info(f"Forcing new file for {picid}")
        with suppress(FileNotFoundError):
            os.remove(csv_file)

    with suppress(FileNotFoundError, ValueError):
        df0 = pd.read_csv(csv_file, index_col=[0])
        logger.success(f"Found existing csv file: {df0}")
        return df0

    logger.debug("Getting Target PSC and subtracting bias")
    psc0 = get_psc(picid, stamps, **kwargs) - camera_bias
    logger.debug(f"Target PSC shape: {psc0.shape}")
    num_frames = psc0.shape[0]

    # Normalize
    logger.debug(f"Normalizing target for {num_frames} frames")
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
                logger.warning(f"Sum for target frame {frame_index} is 0")
        except RuntimeWarning:
            logger.warning(f"Skipping frame {frame_index}")

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
                logger.debug(f"Skipping PICID {source_index}, low snr {snr:.02f}")
                continue
        except KeyError:
            logger.debug(f"No source in table: {picid}")
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
        data = dict()
        try:
            v = ((normalized_psc0 - normalized_psc1) ** 2).sum()
            data[source_index] = v
        except ValueError as e:
            logger.debug(f"Skipping invalid stamp for source {source_index}: {e!r}")

    df0 = pd.DataFrame(
        {'v': list(data.values())},
        index=list(data.keys())).sort_values(by='v')

    if csv_file:
        df0[:num_refs].to_csv(csv_file)

    return df0


def get_psc(picid, stamps, frame_slice=None):
    try:
        psc = stamps.query('picid == @picid').filter(like='pixel_').to_numpy()
    except KeyError:
        raise Exception(f"{picid} not found in the stamp collection.")

    if frame_slice is not None:
        psc = psc[frame_slice]

    return psc


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


def get_stamp_size(df0, superpixel_padding=1):
    """Get the stamp size for given pixel drifts.

    This will find the median drift length in both coordinate axes and append
    a padding of superpixels. The returned length is for an assumed square postage
    stamp.

    Example:

        Here the star coords (given by `.`) drift an average of 2 pixels in the
        x-direction and 7 pixels in the y-direction. Since 7 is the larger of
        the two it is used as the base, which is rounded up to the nearest number
        of superpixels, so a stamp that is 8x8 pixels (represented by `o`). We
        then add a default of one superpixel padding (`x`) around the stamp to
        give 8+(2+2)=12

                gbxxxxxxxxgb
                rgxxxxxxxxrg
                xxooo..oooxx
                xxooo..oooxx
                xxoo..ooooxx
                xxooo..oooxx
                xxooo..oooxx
                xxooo..oooxx
                xxooo..oooxx
                xxooooooooxx
                gbxxxxxxxxgb
                rgxxxxxxxxrg

    Args:
        df0 (`pandas.DataFrame`): A DataFrame that includes the `x_max/x_min` and
            `y_max/y_min` columns
        superpixel_padding (int, optional): The number of superpixels to place
            around the area the star traverses.

    Returns:
        int: The length of one side of a square postage stamp.
    """
    # Get the movement stats
    x_range_mean, x_range_med, x_range_std = sigma_clipped_stats(df0.x_max - df0.x_min)
    y_range_mean, y_range_med, y_range_std = sigma_clipped_stats(df0.y_max - df0.y_min)

    # Get the larger of the two movements
    stamp_size = max(int(x_range_med + round(x_range_std)), int(y_range_med + round(y_range_std)))

    # Round to nearest superpixel integer
    stamp_size = 2 * round(stamp_size / 2)

    # Add padding for our superpixels (i.e. number of superpixels * pixel width of superpixel)
    stamp_size += (superpixel_padding * 4)

    return stamp_size


def get_postage_stamps(point_sources,
                       fits_fn,
                       output_fn=None,
                       stamp_size=10,
                       x_column='measured_x',
                       y_column='measured_y',
                       force=False):
    """Extract postage stamps for each PICID in the given file.

    Args:
        point_sources (`pandas.DataFrame`): A DataFrame containing the results from `source-extractor`.
        fits_fn (str): The name of the FITS file to extract stamps from.
        output_fn (str, optional): Path for output csv file.
        stamp_size (int, optional): The size of the stamp to extract, default 10 pixels.
        x_column (str): The name of the column to use for the x position.
        y_column (str): The name of the column to use for the y position.
        force (bool): If should create new file if old exists, default False.
    Returns:
        str: The path to the csv file with
    """

    row = point_sources.iloc[0]
    output_fn = output_fn or f'{row.unit_id}-{row.camera_id}-{row.seq_time}-{row.img_time}.csv'
    logger.debug(f'Looking for {output_fn=}')
    if os.path.exists(output_fn) and force is False:
        logger.info(f'{output_fn} already exists and force=False, returning')
        return output_fn

    data = fits.getdata(fits_fn)

    logger.debug(f'Extracting {len(point_sources)} point sources from {fits_fn}')

    logger.debug(f'Starting source extraction for {fits_fn}')
    with open(output_fn, 'w') as metadata_fn:
        writer = csv.writer(metadata_fn, quoting=csv.QUOTE_MINIMAL)

        # Write out headers.
        csv_headers = [
            'picid',
            'unit_id',
            'camera_id',
            'time',
            'slice_y_start',
            'slice_y_stop',
            'slice_x_start',
            'slice_x_stop',
        ]
        csv_headers.extend([f'pixel_{i:02d}' for i in range(stamp_size ** 2)])
        writer.writerow(csv_headers)

        for idx, row in point_sources.iterrows():
            # Get the stamp for the target
            target_slice = bayer.get_stamp_slice(
                row[x_column], row[y_column],
                stamp_size=(stamp_size, stamp_size),
                ignore_superpixel=False
            )

            # Add the target slice to metadata to preserve original location.
            row['target_slice'] = target_slice
            stamp = data[target_slice].flatten().tolist()

            row_values = [
                int(row.picid),
                str(row.unit_id),
                str(row.camera_id),
                row.time,
                target_slice[0].start,
                target_slice[0].stop,
                target_slice[1].start,
                target_slice[1].stop,
                *stamp
            ]

            # Write out stamp data
            writer.writerow(row_values)

    logger.debug(f'PSC file saved to {output_fn}')
    return output_fn
