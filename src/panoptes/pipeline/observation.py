from typing import List
from urllib.error import HTTPError

import numpy.typing as npt
import pandas
import pandas as pd
from google.cloud import firestore
from panoptes.utils.images import bayer
from pydantic import BaseSettings
from loguru import logger


class Settings(BaseSettings):
    COLUMN_X: str = 'catalog_wcs_x'
    COLUMN_Y: str = 'catalog_wcs_y'


settings = Settings()
db = firestore.Client()


def get_stamp_locations(sources_file_list: List[str]) -> pandas.DataFrame:
    """Get xy pixel locations for each source in an observation."""
    print(f'Getting {len(sources_file_list)} remote files.')
    position_dfs = list()
    for url in sources_file_list:
        try:
            pos_df = pd.read_parquet(url, columns=[settings.COLUMN_X, settings.COLUMN_Y])
            position_dfs.append(pos_df)
        except HTTPError as e:
            logger.warning(f'Problem loading parquet at {url=} {e!r}')

    num_frames = len(position_dfs)
    print(f'Combining {num_frames} position files')
    catalog_positions = pd.concat(position_dfs).sort_index()
    print(f'Loaded a total of {len(catalog_positions)}')

    # Filter the sources that weren't detected in all frames.
    # TODO in the future we could process all sources from a catalog.
    counts = catalog_positions.reset_index().groupby('picid').count()
    print(f'Filtering to sources that appear in all {num_frames} frames')
    catalog_positions = catalog_positions.loc[counts[counts == num_frames].dropna().index]
    print(f'Filtered to {len(catalog_positions)} sources')

    # Make xy catalog with the average positions from all measured frames.
    xy_catalog = catalog_positions.reset_index().groupby('picid')

    # Get max diff in xy positions.
    x_catalog_diff = (xy_catalog.catalog_wcs_x.max() - xy_catalog.catalog_wcs_x.min()).max()
    y_catalog_diff = (xy_catalog.catalog_wcs_y.max() - xy_catalog.catalog_wcs_y.min()).max()

    if x_catalog_diff >= 18 or y_catalog_diff >= 18:
        raise RuntimeError(f'Too much drift! {x_catalog_diff=} {y_catalog_diff}')

    stamp_width = 10 if x_catalog_diff < 10 else 18
    stamp_height = 10 if y_catalog_diff < 10 else 18

    # Determine stamp size
    stamp_size = (stamp_width, stamp_height)
    print(f'Using {stamp_size=}.')

    # Get the mean positions
    xy_mean = xy_catalog.mean()
    xy_std = xy_catalog.std()

    xy_mean = xy_mean.rename(columns=dict(
        catalog_wcs_x=f'{settings.COLUMN_X}_mean',
        catalog_wcs_y=f'{settings.COLUMN_Y}_mean')
    )
    xy_std = xy_std.rename(columns=dict(
        catalog_wcs_x=f'{settings.COLUMN_X}_std',
        catalog_wcs_y=f'{settings.COLUMN_Y}_std')
    )

    xy_mean = xy_mean.join(xy_std)

    stamp_positions = xy_mean.apply(
        lambda row: bayer.get_stamp_slice(row[f'{settings.COLUMN_X}_mean'],
                                          row[f'{settings.COLUMN_Y}_mean'],
                                          stamp_size=stamp_size,
                                          as_slices=False,
                                          ), axis=1, result_type='expand')

    stamp_positions[f'{settings.COLUMN_X}_mean'] = xy_mean[f'{settings.COLUMN_X}_mean']
    stamp_positions[f'{settings.COLUMN_Y}_mean'] = xy_mean[f'{settings.COLUMN_Y}_mean']
    stamp_positions[f'{settings.COLUMN_X}_std'] = xy_mean[f'{settings.COLUMN_X}_std']
    stamp_positions[f'{settings.COLUMN_Y}_std'] = xy_mean[f'{settings.COLUMN_Y}_std']

    stamp_positions.rename(columns={0: 'stamp_y_min',
                                    1: 'stamp_y_max',
                                    2: 'stamp_x_min',
                                    3: 'stamp_x_max'}, inplace=True)

    return stamp_positions


def make_stamps(stamp_positions: pandas.DataFrame,
                data: npt.DTypeLike,
                ) -> pandas.DataFrame:
    stamp_width = int(stamp_positions.stamp_x_max.mean() - stamp_positions.stamp_x_min.mean())
    stamp_height = int(stamp_positions.stamp_y_max.mean() - stamp_positions.stamp_y_min.mean())
    total_stamp_size = int(stamp_width * stamp_height)
    logger.debug(
        f'Making stamps of {total_stamp_size=} for {len(stamp_positions)} sources from data {data.shape}')

    stamps = []
    for picid, row in stamp_positions.iterrows():
        # Get the stamp data.
        row_slice = slice(int(row.stamp_y_min), int(row.stamp_y_max))
        col_slice = slice(int(row.stamp_x_min), int(row.stamp_x_max))
        psc0 = data[row_slice, col_slice].reshape(-1)

        # Make sure stamp is correct size (errors at edges).
        if psc0.shape == (total_stamp_size,):
            stamp = pd.DataFrame(psc0).T
            stamp.columns = [f'pixel_{i:03d}' for i in range(total_stamp_size)]
            stamp['picid'] = picid
            stamp.set_index(['picid'], inplace=True)
            stamps.append(stamp)

    # Make one dataframe.
    psc_data = pd.concat(stamps).sort_index()

    return psc_data
