import os
import csv
import shutil
from enum import IntEnum

import pandas as pd
import numpy as np
from panoptes.utils.images.bayer import get_rgb_background
from scipy import ndimage
from sklearn.linear_model import LinearRegression

from tqdm.auto import tqdm

from astropy.io import fits

from panoptes.utils import listify
from panoptes.utils.images import bayer
from panoptes.utils.logging import logger
from panoptes.utils.images import fits as fits_utils

from panoptes.pipeline.utils import sources
from panoptes.pipeline.utils.gcp.bigquery import get_bq_clients
from panoptes.pipeline.utils import metadata


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

    for fn in tqdm(stamp_files, desc='Loading stamp files'):
        df0 = pd.read_csv(fn)
        if picid is not None:
            df0 = df0.set_index('picid').loc[picid].reset_index()

        stamps.append(df0)

    stamps = pd.concat(stamps)
    stamps.picid = stamps.picid.astype('int')
    stamps.time = pd.to_datetime(stamps.time, utc=True)

    return stamps.sort_values(by='time')


def get_stamp_info(df0, range_sigma=1, superpixel_size=4):
    """Get the stamp positions and size for given pixel drifts.

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
        df0 (`pandas.DataFrame`): A DataFrame that includes the 'catalog_wcs_x_int'
            and 'catalog_wcs_y_int' columns.

    Returns:
        pandas.DataFrame: DataFrame with mean catalog positions and a calculated
            stamp_size.
    """
    # Make xy catalog with the average positions from all measured frames.
    xy_catalog = df0.reset_index().filter(
        regex='picid|^catalog_wcs_x_int$|^catalog_wcs_y_int$')

    # Get just the position columns
    xy_mean = xy_catalog.groupby('picid').mean()

    # Get the range for the measured peaks
    xy_range = xy_catalog.groupby('picid').transform(lambda grp: grp.max() - grp.min())
    xy_range.rename(columns=dict(catalog_wcs_x_int='x_range', catalog_wcs_y_int='y_range'),
                    inplace=True)
    xy_range['picid'] = df0.picid

    # Add the range columns
    xy_mean = xy_mean.join(xy_range.groupby('picid').max())

    # Get the upper quantile of all sources plus one std
    stamp_range = np.ceil(
        max((xy_mean.quantile(q=0.75) + (range_sigma * xy_mean.std())).filter(regex='range')))
    # Rounds up to the nearest allowed size based on the superpixel.
    add_pixels = ((superpixel_size - (stamp_range - 2)) % superpixel_size)
    stamp_size = int(stamp_range + add_pixels)

    xy_mean['stamp_size'] = stamp_size

    return xy_mean.reset_index()


def get_postage_stamps(point_sources,
                       fits_fn,
                       output_fn=None,
                       stamp_size=10,
                       x_column='catalog_wcs_x_int',
                       y_column='catalog_wcs_y_int',
                       global_background=True,
                       rgb_background=None,
                       camera_bias=0,
                       force=False,
                       show_progress=True,
                       ):
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
    data = data - camera_bias
    image_time = pd.to_datetime(os.path.basename(fits_fn).split('.')[0])
    unit_id, camera_id, seq_time = header['SEQID'].split('_')

    output_fn = output_fn or f'{unit_id}-{camera_id}-{seq_time}-{image_time}.csv'

    logger.debug(f'Looking for output file {output_fn}')
    if os.path.exists(output_fn) and force is False:
        logger.debug(f'{output_fn} already exists and force=False, returning existing file')
        return output_fn

    if global_background and rgb_background is None:
        logger.debug(f'Getting global RGB background')
        rgb_background = bayer.get_rgb_background(fits_fn)

    logger.debug(f'Extracting {len(point_sources)} point sources from {fits_fn}')
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

        iter = point_sources.iterrows()
        if show_progress:
            iter = tqdm(iter, total=len(point_sources), desc='Making stamps')

        for idx, row in iter:
            # Get the stamp for the target
            target_slice = bayer.get_stamp_slice(
                row[x_column], row[y_column],
                stamp_size=(stamp_size, stamp_size),
                ignore_superpixel=False
            )

            # Get the stamp from the data and subtract the camera bias.
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

            if global_background and rgb_background is not None:
                # Subtract the RGB background, which should not include the bias.
                stamp = stamp - rgb_background[target_slice]

            row_values.extend(stamp.flatten().tolist())

            # Write out stamp data
            writer.writerow(row_values)

    logger.debug(f'PSC file saved to {output_fn}')
    return output_fn


def process_observation_sources(sequence_id,
                                output_dir='.',
                                num_refs=200,
                                camera_bias=2048,
                                saturation_limit=11495,  # After subtracting bias.
                                bg_box_size=(79, 84),
                                observation_filename='observation.parquet',
                                force_new=False,
                                frame_slice=slice(None, None),
                                ):
    logger.info(f'Processing {sequence_id}')

    sequence_dir = os.path.realpath(f'{output_dir}/{sequence_id}')

    # Cleanup old processing if requested.
    if force_new and os.path.exists(sequence_dir):
        logger.info(f'Forcing new processing by deleting existing {sequence_dir}')
        shutil.rmtree(sequence_dir, ignore_errors=True)

    # Create directories for output.
    subdir_names = ['sources', 'lightcurves', 'psc', 'stamps', 'images']
    subdirs = dict()
    for subdir in subdir_names:
        dir_path = f'{sequence_dir}/{subdir}'
        subdirs[subdir] = dir_path
        os.makedirs(dir_path, exist_ok=True)

    # Get the metadata from local file.
    images_df = metadata.get_metadata(sequence_id=sequence_id).sort_values(by='time')

    # Get the list of local paths for the downloaded images.
    image_list = images_df.public_url.to_list()

    # Download the files
    fits_files = metadata.download_images(image_list, subdirs['images'])[frame_slice]
    num_frames = len(fits_files)

    # Lookup the stars in the field.
    catalog_stars_fn = f'{sequence_dir}/catalog-stars.parquet'
    try:
        catalog_stars_df = pd.read_parquet(catalog_stars_fn)
    except (FileNotFoundError, OSError):
        # BigQuery lookup.
        bq_client, bqstorage_client = get_bq_clients()

        # Get the WCS for the middle image. This searches with a larger radius than
        # the WCS so only one lookup is needed even if all the frames are slightly off.
        wcs0 = fits_utils.getwcs(fits_files[int(num_frames / 2)])
        catalog_stars_df = sources.get_stars_from_wcs(wcs0,
                                                      vmag_min=6,
                                                      vmag_max=13,
                                                      # numcont=5,
                                                      bq_client=bq_client,
                                                      bqstorage_client=bqstorage_client)
        catalog_stars_df.to_parquet(catalog_stars_fn, index=False)

    # Get the xy positions of the catalog stars for each frame.
    all_point_sources = list()
    for fits_file in tqdm(fits_files, desc='Getting star positions'):
        image_id = fits_utils.getval(fits_file, 'IMAGEID')
        sources_filename = f'{subdirs["sources"]}/{image_id}-metadata.parquet'

        try:
            if not os.path.exists(sources_filename):
                # Get the stellar positions for all catalog stars using WCS for frame.
                wcs0 = fits_utils.getwcs(fits_file)
                point_sources = sources.get_xy_positions(wcs0, catalog_stars_df)

                # Get unit and camera id. Time from the filename.
                unit_id, camera_id, _ = image_id.split('_')
                obstime = os.path.splitext(os.path.basename(fits_file))[0]

                # Add the id columns.
                point_sources['unit_id'] = unit_id
                point_sources['camera_id'] = camera_id
                point_sources['time'] = obstime

                if point_sources is not None:
                    point_sources.to_parquet(sources_filename, index=False)
                    all_point_sources.append(point_sources)
            else:
                # Load existing files.
                all_point_sources.append(pd.read_parquet(sources_filename))
        except Exception as e:
            tqdm.write(f'Error: {fits_file} {e!r}')

    # Make a single dataframe for all the frames.
    obs_sources_df = pd.concat(all_point_sources)
    obs_sources_df.convert_dtypes(convert_integer=False)
    obs_sources_df.time = pd.to_datetime(obs_sources_df.time, utc=True)
    del all_point_sources

    image_columns_drop = [
        'bucket_path',
        'local_file',
        'sequence_id',
        'image_id'
    ]
    images_df0 = images_df.drop(columns=image_columns_drop, errors='ignore')
    images_df0.time = pd.to_datetime(images_df0.time, utc=True)

    # Merge individual image metadata with full observation metadata.
    obs_sources_df = obs_sources_df.merge(images_df0, on=['time', 'unit_id'])
    del images_df0

    # Figure out the appropriate stamp size.
    xy_mean = get_stamp_info(obs_sources_df)
    stamp_size = xy_mean['stamp_size'].iloc[0]

    # Get slices
    image_slice_files = list()
    for fits_file in tqdm(fits_files,
                          desc=f'Making {stamp_size}x{stamp_size} stamps'):
        image_time = os.path.splitext(os.path.basename(fits_file))[0]
        stamp_fn = f'{subdirs["stamps"]}/{image_time}-stamps.csv'

        if not os.path.exists(stamp_fn):
            # Get and save the RGB background, which will include the camera bias.
            rgb_background = lookup_rgb_background(fits_file,
                                                   box_size=bg_box_size,
                                                   camera_bias=camera_bias,
                                                   save=True)
            csv_fn = get_postage_stamps(xy_mean,
                                        fits_file,
                                        stamp_size=stamp_size,
                                        x_column='catalog_wcs_x_int',
                                        y_column='catalog_wcs_y_int',
                                        output_fn=stamp_fn,
                                        global_background=True,
                                        rgb_background=rgb_background,
                                        camera_bias=camera_bias,
                                        force=force_new,
                                        show_progress=False
                                        )
            image_slice_files.append(csv_fn)
        else:
            image_slice_files.append(stamp_fn)

    full_info_fn = f'{sequence_dir}/{observation_filename}'

    # Set the index
    kdims = ['unit_id', 'camera_id', 'picid', 'time']

    # Load all the stamps.
    if not os.path.exists(full_info_fn):
        logger.info(f'Loading Postage Stamp Cubes (PSC)')
        stamps = load_stamps(image_slice_files)
        stamps.time = pd.to_datetime(stamps.time, utc=True)
        stamps_df = stamps.set_index(kdims).sort_index()

        obs_sources_df = obs_sources_df.set_index(kdims).sort_index().merge(stamps_df,
                                                                            left_index=True,
                                                                            right_index=True)
        obs_sources_df.to_parquet(full_info_fn)
    else:
        obs_sources_df = pd.read_parquet(full_info_fn)

    # Normalize stamp data
    logger.info(f'Normalizing PSCs')
    psc_data = obs_sources_df.filter(regex='pixel_')
    norm_psc_data = (psc_data.T / psc_data.sum(1)).T

    num_frames = len(obs_sources_df.reset_index().time.unique())
    picid_list = list(obs_sources_df.index.get_level_values('picid').unique())

    logger.info(f'Processing {len(picid_list)} stars over {num_frames} frames.')
    for picid in tqdm(picid_list, desc='Looking at stars'):
#         tqdm.write(f'Starting {picid}')
        lightcurve_path = f'{subdirs["lightcurves"]}/{picid}.csv'
        psc_path = f'{sequence_dir}/psc/{picid}.npz'

        if os.path.exists(lightcurve_path) and os.path.exists(psc_path):
            continue

        # Get the target data
        target_norm_psc_data = norm_psc_data.loc[:, :, picid, :]

        # Get the SSD for each stamp separately.
#         tqdm.write(f'Making ssd per frame for {picid}')
        ssd_per_frame = norm_psc_data.rsub(target_norm_psc_data).pow(2).sum(1)

        # The sum of the SSDs for each source give final score, with smaller values better.
        # The top value should have a score of `0` and should be the target.
        final_scores = ssd_per_frame.sum(level='picid').sort_values()

        assert final_scores.iloc[0] == 0.
#         tqdm.write('Have final scores for {picid}')

        # Get the top refs by score. Skip the target at index=0.
        top_refs_list = final_scores[1:num_refs + 1].index.tolist()

        # Filter the observation dataframe down to just the references.
        refs_meta_index = obs_sources_df.index.get_level_values('picid').isin(top_refs_list)
        refs_df = obs_sources_df[refs_meta_index].copy().sort_index()

        # Get SSD value for references for each frame.
        ssd_index = ssd_per_frame.index.get_level_values('picid').isin(top_refs_list)
        refs_scores = ssd_per_frame[ssd_index].reset_index()
        refs_scores.time = pd.to_datetime(refs_scores.time, utc=True)

        # Add final score to the reference metadata.
        refs_df['similarity_score'] = refs_scores.set_index(kdims)

        # ### Create comparison star

        # Get PSCs for target and references.
        drop_index_cols = ['unit_id', 'camera_id']
        refs_psc = psc_data[refs_meta_index].droplevel(drop_index_cols)
        norm_refs_psc = norm_psc_data[refs_meta_index].droplevel(drop_index_cols)

        # Build the comparison PSC.
#         tqdm.write('Building comparison star for {picid}')
        comparison_psc = build_comparison_psc(norm_refs_psc, num_refs, refs_psc,
                                              target_norm_psc_data)

        # Get target array of same size/shape as comparison.
        target_psc_data = psc_data.loc[:, :, picid, :].droplevel(drop_index_cols)

        # Put into an actual cube for RGB extraction.
        comparison_psc = comparison_psc.reshape(num_frames, stamp_size, stamp_size)
        target_psc = target_psc_data.to_numpy().reshape(num_frames, stamp_size, stamp_size)

        # Split the data into three colors.
#         target_rgb_psc = bayer.get_rgb_data(target_psc)
#         comp_rgb_psc = bayer.get_rgb_data(comparison_psc)

        # Get background subtracted target and comparison using global background.
        # Mask negative pixels.
#         target_rgb_psc = np.ma.masked_less(target_rgb_psc, 0)
#         comp_rgb_psc = np.ma.masked_less(comp_rgb_psc, 0)

        # Mask saturated pixels.
#         target_rgb_psc = np.ma.masked_greater(target_rgb_psc, saturation_limit)
#         comp_rgb_psc = np.ma.masked_greater(comp_rgb_psc, saturation_limit)

#         target_rgb_psc = mask_outliers(target_rgb_psc, upper_limit=saturation_limit)
#         comp_rgb_psc = mask_outliers(comp_rgb_psc, upper_limit=saturation_limit)

#         target_df = obs_sources_df.query('picid==@picid').reset_index().sort_values(by='time')

        logger.info('Saving final PSC for {picid}')
        np.savez_compressed(psc_path,
                            target=target_psc.astype('int16'),
                            comparison=comparison_psc.astype('int16'))

#         final_df = pd.DataFrame({'target': target_psc.reshape(num_frames, -1),
#                                  'comparison': comparison_psc.reshape(num_frames, -1)},
#                                index=target_df.time)
#         final_df['picid'] = picid
#         final_df.reset_index().to_csv(lightcurve_path, index=False)

        # # Make apertures from the comparison star.
        # apertures = make_apertures(comp_rgb_psc)
        #
        # # Save the target and comp PSC.
        # np.savez_compressed(psc_path, target=target_rgb_psc.data,
        #                     comparison=comp_rgb_psc.data,
        #                     aperture=apertures)

        # rgb_lc_df = make_rgb_lightcurve(target_rgb_psc,
        #                                 comp_rgb_psc,
        #                                 apertures,
        #                                 target_time,
        #                                 norm=True).reset_index()
        # rgb_lc_df['picid'] = picid
        # rgb_lc_df.to_csv(lightcurve_path, index=False)


def build_comparison_psc(norm_refs_psc, num_refs, refs_psc, target_norm_psc_data):
    X_train = norm_refs_psc.to_numpy().reshape(num_refs, -1).T
    X_test = refs_psc.to_numpy().reshape(num_refs, -1).T

    y_train = target_norm_psc_data.to_numpy().flatten()
    # y_test = target_psc_data.to_numpy().flatten()

    lin_reg = LinearRegression().fit(X_train, y_train)

    # Predict comparison from comparison stars with flux.
    comp_psc = lin_reg.predict(X_test)

    return comp_psc


def mask_outliers(psc, lower_limit=0, upper_limit=11490):
    rgb = list()
    for color in RGB:
        # Mask negative pixels.
        df0 = np.ma.masked_less(psc[color], lower_limit)

        # Mask saturated pixels.
        df0 = np.ma.masked_greater(df0, upper_limit)

        rgb.append(df0)

    rgb_mask = np.ma.array(rgb)

    return rgb_mask.sum(axis=0)


def make_apertures(rgb_data, smooth_filter=np.ones((3, 3)), *args, **kwargs):
    # Get RGB data and make single stamp.
    apertures = [~ndimage.binary_opening(rgb_data[i],
                                         structure=smooth_filter,
                                         iterations=1) for i in range(len(rgb_data))]
    return np.array(apertures)


def lookup_rgb_background(filename, box_size=(79, 84), camera_bias=0, save=True, **kwargs):
    sub_fn = filename.replace('.fits', '-bg-subtracted.fits')

    if os.path.exists(sub_fn):
        # Get background from existing file, stored in second extension.
        rgb_data = fits_utils.getdata(sub_fn, ext=1)
    else:
        rgb_data = get_rgb_background(filename, box_size=box_size, camera_bias=camera_bias, *kwargs)

        if save:
            sub_data = fits_utils.getdata(filename) - camera_bias - rgb_data

            sub_hdu = fits.PrimaryHDU(sub_data.astype('int16'))
            back_hdu = fits.ImageHDU(rgb_data.astype('int16'))

            hdul = fits.HDUList([sub_hdu, back_hdu])
            hdul.writeto(sub_fn, overwrite=True)

    return rgb_data


def make_rgb_lightcurve(target, comp, apertures, target_time, freq=None, norm=False):
    # Get just aperture data.
    target_rgb_aperture_data = np.ma.array(target.data, mask=apertures)
    comp_rgb_aperture_data = np.ma.array(comp.data, mask=apertures)

    # Split aperture data into rgb.
    target_rgb_aperture = bayer.get_rgb_data(target_rgb_aperture_data)
    comp_rgb_aperture = bayer.get_rgb_data(comp_rgb_aperture_data)

    lcs = dict()
    num_frames = len(target_time)
    for color in RGB:
        target_sum = target_rgb_aperture[color].reshape(num_frames, -1).sum(1)
        comp_sum = comp_rgb_aperture[color].reshape(num_frames, -1).sum(1)
        lc = target_sum / comp_sum
        lc_df = pd.DataFrame(lc, index=target_time)

        if freq is not None:
            lc_df = lc_df.resample(freq).mean()

        # The 'mean' above names the column '0'.
        lc_values = lc_df[0].values
        if norm:
            lc_values = lc_values / np.ma.median(lc_values)

        lcs[color.name.lower()] = lc_values

    df = pd.DataFrame(lcs, index=lc_df.index)

    return df
