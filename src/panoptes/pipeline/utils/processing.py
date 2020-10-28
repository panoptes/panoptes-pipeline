import os
import csv

import numpy as np
import pandas as pd
from panoptes.utils import listify
from scipy import linalg

from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from panoptes.utils.images import bayer
from panoptes.utils.logging import logger

from tqdm import tqdm


def find_similar_stars(
        picid,
        stamp_files,
        num_refs=200,
        show_progress=True,
        *args, **kwargs):
    """ Find PSCs in stamps that are morphologically similar to the PSC for picid.

    Args:
        stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
        (int): Index of target PIC
    """
    logger.info(f"Finding similar stars for PICID {picid}")

    stamp_files = listify(stamp_files)

    iterator = stamp_files
    if show_progress:
        iterator = tqdm(iterator, total=len(stamp_files))

    refs_list = list()
    for fits_file in iterator:
        # Load all the stamps and metadata
        stamps0 = pd.read_csv(fits_file)
        stamps0.picid = stamps0.picid.astype('int')

        # Get the target.
        target_psc = stamps0.query('picid==@picid')

        # Get just the stamp data minus the camera bias.
        stamp_data = stamps0.filter(regex='pixel').to_numpy()
        target_stamp = target_psc.filter(regex='pixel').to_numpy()

        # Normalize stamps.
        normalized_stamps = (stamp_data.T / stamp_data.sum(1)).T
        normalized_target_stamp = target_stamp / target_stamp.sum()

        # Get the summed squared difference between stamps and target.
        loss_score = ((normalized_stamps - normalized_target_stamp) ** 2).sum(1)

        # Return sorted score series.
        sort_idx = np.argsort(loss_score)
        refs_list.append(pd.Series(loss_score[sort_idx], index=stamps0.picid[sort_idx]))

    # Combine all the loss scores.
    target_refs = pd.concat(refs_list).reset_index().rename(columns={0: 'score'})

    # Group by star and return top sorted scores.
    top_refs = target_refs.groupby('picid').sum().sort_values(by='score')[:num_refs+1]

    return top_refs


def get_refs_for_file(fits_file, stamps, num_refs=200):
    """Find references for all stamps. Not super efficient."""
    stamps0 = pd.read_csv(fits_file)
    picids = stamps0.picid.values

    stamp_data = stamps0.filter(regex='pixel').to_numpy()
    normalized_stamps = (stamp_data.T / stamp_data.sum(1)).T

    refs_list = list()
    for i, normalized_target_stamp in tqdm(enumerate(normalized_stamps), total=len(picids)):
        refs_list.append(picids[np.argsort(((normalized_stamps - normalized_target_stamp) ** 2).sum(1))[:num_refs]])

    return refs_list


def load_stamps(stamp_files, picid=None):
    """Load the stamp files with optional PICID filter."""
    stamps = list()

    for fn in tqdm(stamp_files):
        df0 = pd.read_csv(fn)
        if picid is not None:
            df0 = df0.set_index('picid').loc[picid].reset_index()

        stamps.append(df0)

    stamps = pd.concat(stamps)
    stamps.picid = stamps.picid.astype('int')
    stamps.time = pd.to_datetime(stamps.time)

    return stamps


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

    The `point_sources` DataFrame should contain the `picid` and
    x_column and y_column specified. The `IMAGEID` in the fits file
    will be used to look up the `unit_id`, `camera_id`, and `time`.

    Args:
        point_sources (`pandas.DataFrame`): A DataFrame containing the results
            the `picid` and x and y columns.
        fits_fn (str): The name of the FITS file to extract stamps from.
        output_fn (str, optional): Path for output csv file, defaults to local
            csv file with id as name.
        stamp_size (int, optional): The size of the stamp to extract, default 10 pixels.
        x_column (str): The name of the column to use for the x position.
        y_column (str): The name of the column to use for the y position.
        force (bool): If should create new file if old exists, default False.
    Returns:
        str: The path to the csv file with
    """
    data, header = fits.getdata(fits_fn, header=True)
    image_time = os.path.basename(fits_fn).split('.')[0]
    unit_id, camera_id, seq_time = header['SEQID'].split('_')

    row = point_sources.iloc[0]
    output_fn = output_fn or f'{unit_id}-{camera_id}-{seq_time}-{image_time}.csv'

    logger.debug(f'Looking for output file {output_fn}')
    if os.path.exists(output_fn) and force is False:
        logger.info(f'{output_fn} already exists and force=False, returning')
        return output_fn

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
                str(unit_id),
                str(camera_id),
                image_time,
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
