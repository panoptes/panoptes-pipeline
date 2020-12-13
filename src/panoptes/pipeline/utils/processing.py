import os
import csv
import glob
from enum import IntEnum

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from scipy import linalg

from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from panoptes.utils import listify
from panoptes.utils.images import bayer
from panoptes.utils.logging import logger

import pandas as pd
import numpy as np
from scipy import ndimage
from sklearn.linear_model import LinearRegression


class RGB(IntEnum):
    """Helper class for array index access."""
    RED = 0
    R = 0
    GREEN = 1
    G = 1
    G1 = 1
    BLUE = 2
    B = 2


def find_similar_stars(
        picid,
        stamp_files,
        num_refs=200,
        show_progress=True,
        *args, **kwargs):
    """ Find PSCs in stamps that are morphologically similar to the PSC for picid.

    Args:
        stamps (np.array): Collection of stamps with axes: frame, PIC, pixels
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
        stamps0 = pd.read_csv(fits_file).sort_values(by='time')

        # Get the target.
        target_psc = stamps0.query('picid==@picid')

        # Get just the stamp data minus the camera bias.
        stamp_data = stamps0.filter(regex='pixel').to_numpy()
        target_stamp = target_psc.filter(regex='pixel').to_numpy()

        # Normalize stamps.
        normalized_stamps = (stamp_data.T / stamp_data.sum(1)).T
        normalized_target_stamp = target_stamp / target_stamp.sum()

        # Get the summed squared difference between stamps and target.
        loss_score = ((normalized_target_stamp - normalized_stamps) ** 2).sum(1)

        # Return sorted score series.
        sort_idx = np.argsort(loss_score)
        df_idx = pd.Index(stamps0.iloc[sort_idx].picid, name='picid')

        refs_list.append(pd.Series(loss_score[sort_idx], index=df_idx))

    # Combine all the loss scores.
    target_refs = pd.concat(refs_list).reset_index().rename(columns={0: 'score'})

    if num_refs is not None:
        # Group by star and return top sorted scores.
        top_refs = target_refs.groupby('picid').sum().sort_values(by='score')[:num_refs + 1]

        return top_refs
    else:
        return target_refs


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

    return stamps.sort_values(by='time')


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
                       x_column='catalog_wcs_x_int',
                       y_column='catalog_wcs_y_int',
                       global_background=True,
                       rgb_background=None,
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
    image_time = pd.to_datetime(os.path.basename(fits_fn).split('.')[0])
    unit_id, camera_id, seq_time = header['SEQID'].split('_')

    output_fn = output_fn or f'{unit_id}-{camera_id}-{seq_time}-{image_time}.csv'

    logger.debug(f'Looking for output file {output_fn}')
    if os.path.exists(output_fn) and force is False:
        logger.info(f'{output_fn} already exists and force=False, returning')
        return output_fn

    if global_background and rgb_background is None:
        logger.debug(f'Getting global RGB background')
        rgb_background = bayer.get_rgb_background(fits_fn)

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

        if global_background:
            csv_headers.extend([
                'background_r',
                'background_g',
                'background_b',
            ])
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
            stamp = data[target_slice]

            # Make sure we got a full stamp.
            if stamp.shape != (stamp_size, stamp_size):
                continue

            row_values = [
                int(row.picid),
                str(unit_id),
                str(camera_id),
                image_time,
                target_slice[0].start,
                target_slice[0].stop,
                target_slice[1].start,
                target_slice[1].stop,
            ]

            if global_background:
                # Get the background and average.
                stamp_rgb_bg = rgb_background[target_slice]
                try:
                    rgb_bg_avg = [int(x.mean()) for x in bayer.get_rgb_data(stamp_rgb_bg)]
                except Exception:
                    rgb_bg_avg = [0, 0, 0]
                finally:
                    row_values.extend(rgb_bg_avg)

            row_values.extend(stamp.flatten().tolist())

            # Write out stamp data
            writer.writerow(row_values)

    logger.debug(f'PSC file saved to {output_fn}')
    return output_fn


def process_observation_sources(sequence_dir,
                                num_refs=200,
                                camera_bias=0,
                                background_sigma=3,
                                observation_filename='observation.parquet',
                                ):
    # Get sequence info.
    os.makedirs(f'{sequence_dir}/lightcurves', exist_ok=True)

    # Load stamps and get list of fits files.
    obs_stamps = pd.concat([pd.read_csv(f) for f in glob.glob(f'{sequence_dir}/sources/*-stamps.csv')])
    kdims = ['unit_id', 'camera_id', 'picid', 'time']
    obs_stamps.time = pd.to_datetime(obs_stamps.time, utc=True)
    obs_stamps = obs_stamps.set_index(kdims).sort_index()

    # Stamp data
    psc_data = obs_stamps.filter(regex='pixel_')
    # Normalize stamp data
    norm_psc_data = (psc_data.T / psc_data.sum(1)).T
    stamp_size = int(np.sqrt(psc_data.shape[-1]))

    # Load metadata for observation.
    obs_df = pd.read_parquet(os.path.join(sequence_dir, observation_filename))

    num_frames = len(obs_stamps.reset_index().time.unique())

    picid_list = obs_stamps.picid.unique()

    for picid in tqdm(picid_list, desc='Looking at stars...'):
        target_df = obs_df.query('picid==@picid')
        lightcurve_path = f'{sequence_dir}/lightcurves/{picid}.csv'

        # Get the target data
        target_psc_data = psc_data.loc[:, :, picid, :]
        target_norm_psc_data = norm_psc_data.loc[:, :, picid, :]

        # Get the SSD for each stamp separately.
        all_refs_frame_scores = norm_psc_data.rsub(target_norm_psc_data).pow(2).sum(1)

        # The sum of the SSDs for each source give final score, with smaller values better.
        # The top value should have a score of `0` and should be the target.
        all_ref_scores = all_refs_frame_scores.sum(level='picid').sort_values()

        # Get the top refs by score. Skip the target at index=0.
        top_refs_list = all_ref_scores[1:num_refs + 1].index.tolist()

        # Filter the observation dataframe down to just the references.
        refs_df = obs_df[obs_df.index.get_level_values(2).isin(top_refs_list)].sort_index()

        # Get frame scores for references.
        ref_picid_list = all_refs_frame_scores.index.get_level_values(3).isin(refs_df.index.get_level_values(2))
        refs_scores = all_refs_frame_scores[ref_picid_list].reset_index()
        refs_scores.time = pd.to_datetime(refs_scores.time, utc=True)

        # Add final score to refs
        refs_df['score'] = refs_scores.set_index(kdims)

        # ### Create comparison star

        # Get PSCs for references
        refs_psc = psc_data[psc_data.index.isin(refs_df.index)].droplevel(['unit_id', 'camera_id'])
        norm_refs_psc_data = norm_psc_data[norm_psc_data.index.isin(refs_df.index)].droplevel(['unit_id', 'camera_id'])

        X_train = norm_refs_psc_data.to_numpy().reshape(num_refs, -1).T
        X_test = refs_psc.to_numpy().reshape(num_refs, -1).T

        y_train = target_norm_psc_data.to_numpy().flatten()
        # y_test = target_psc_data.to_numpy().flatten()

        lin_reg = LinearRegression().fit(X_train, y_train)

        # Predict comparison from comparison stars with flux.
        comp_psc = lin_reg.predict(X_test).reshape(num_frames, stamp_size, stamp_size)

        # Get target array of same size/shape as comparison.
        target_psc = target_psc_data.to_numpy().reshape(num_frames, stamp_size, stamp_size)

        # ### Apertures
        # #### Subtract background

        # Split the data into three colors.
        target_rgb_psc = bayer.get_rgb_data(target_psc) - camera_bias
        comp_rgb_psc = bayer.get_rgb_data(comp_psc) - camera_bias

        # Get global background for target.
        target_back_df = obs_stamps.query('picid==@picid').sort_values(by='time').filter(regex='background')
        target_back_df.index = target_df.time
        target_back_df = target_back_df.rename(
            columns={f'background_{c.name.lower()[0]}': f'{c.name.lower()[0]}' for c in RGB})

        # Get background subtracted target and comparison using global background.
        rgb_bg_sub_data = get_rgb_bg_sub_mask(target_rgb_psc, target_back_df.to_numpy().T,
                                              background_sigma=background_sigma).sum(axis=0)
        comp_rgb_bg_sub_data = get_rgb_bg_sub_mask(comp_rgb_psc, target_back_df.to_numpy().T,
                                                   background_sigma=background_sigma).sum(axis=0)

        # Make apertures from the comparison star.
        apertures = make_apertures(comp_rgb_bg_sub_data)

        # Get just aperture data.
        target_rgb_aperture_data = np.ma.array(rgb_bg_sub_data.data, mask=apertures)
        comp_rgb_aperture_data = np.ma.array(comp_rgb_bg_sub_data.data, mask=apertures)

        # Split aperture data into rgb.
        target_rgb_aperture = bayer.get_rgb_data(target_rgb_aperture_data)
        comp_rgb_aperture = bayer.get_rgb_data(comp_rgb_aperture_data)

        rgb_lc_df = make_rgb_lightcurve(target_rgb_aperture, comp_rgb_aperture, target_df.time, norm=True).reset_index()
        rgb_lc_df['picid'] = picid
        rgb_lc_df.to_csv(lightcurve_path, index=False)


def get_rgb_bg_sub_mask(psc, background, background_std=None, background_sigma=3):
    if background_std is None:
        background_std = background.std(1)

    rgb = list()
    for color in RGB:
        sub_back = background[color] + (background_sigma * background_std[color])

        # Subtract from comparison
        sub_comp = (psc[color].T - sub_back).T

        sub_comp = np.ma.masked_less(sub_comp, 0)

        rgb.append(sub_comp)

    rgb_mask = np.ma.array(rgb)

    return rgb_mask


def make_apertures(rgb_data, smooth_filter=np.ones((3, 3)), *args, **kwargs):
    # Get RGB data and make single stamp.
    apertures = [~ndimage.binary_opening(rgb_data[i],
                                         structure=smooth_filter,
                                         iterations=1) for i in range(len(rgb_data))]
    return np.ma.array(apertures)


def make_rgb_lightcurve(target, comp, target_time, freq=None, norm=False):
    lcs = dict()
    num_frames = len(target_time)
    for color in RGB:
        lc = target[color].reshape(num_frames, -1).sum(1) / comp[color].reshape(num_frames, -1).sum(1)
        lc_df = pd.DataFrame(lc, index=target_time)

        if freq is not None:
            lc_df = lc_df.resample(freq).mean()

        lc_values = lc_df[0].values
        if norm:
            lc_values = lc_values / np.ma.median(lc_values)

        lcs[color.name.lower()] = lc_values

    df = pd.DataFrame(lcs, index=lc_df.index)

    return df
