from contextlib import suppress

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import download_file
from dateutil.parser import parse as date_parse
from panoptes.utils import listify
from panoptes.utils.logging import logger
from panoptes.utils.time import current_time
from tqdm import tqdm

OBS_BASE_URL = 'https://storage.googleapis.com/panoptes-observations'


def get_metadata(sequence_id=None, fields=None, show_progress=False):
    """Access PANOPTES data from the network.

    This function is capable of searching one type of object at a time, which is
    specified via the respective id parameter.

    >>> from panoptes.utils.data import get_metadata
    >>> # Get all image metadata for the observation.
    >>> sequence_id = 'PAN001_14d3bd_20170405T100854'
    >>> observation_df = get_metadata(sequence_id=sequence_id)
    >>> type(observation_df)
    <class 'pandas.core.frame.DataFrame'>
    >>> print('Total exptime: ', observation_df.image_exptime.sum())
    Total exptime:  7200.0

    >>> # It's also possible to request certain fields
    >>> airmass_df = get_metadata(sequence_id=sequence_id, fields=['image_airmass'])
    >>> airmass_df.head()
       image_airmass                    sequence_id                      time
    0       1.174331  PAN001_14d3bd_20170405T100854 2017-04-05 10:10:20+00:00
    1       1.182432  PAN001_14d3bd_20170405T100854 2017-04-05 10:13:09+00:00
    2       1.190880  PAN001_14d3bd_20170405T100854 2017-04-05 10:15:59+00:00
    3       1.199631  PAN001_14d3bd_20170405T100854 2017-04-05 10:18:49+00:00
    4       1.208680  PAN001_14d3bd_20170405T100854 2017-04-05 10:21:40+00:00

    Args:
        sequence_id (str|list|None): The list of sequence_ids associated with an observation.
        fields (list|None):  A list of fields to fetch from the database. If None,
            returns all fields.
        show_progress (bool): If True, show a progress bar, default False.

    Returns:

    """
    # Get observation metadata from firestore.
    if sequence_id is not None:
        return get_observation_metadata(sequence_id, fields=fields, show_progress=show_progress)


def get_observation_metadata(sequence_ids, fields=None, show_progress=False):
    """Get the metadata for given sequence_ids.

    Args:
        sequence_ids (list): A list of sequence_ids as strings.
        fields (list|None):  A list of fields to fetch from the database in addition
            to the 'time' and 'sequence_id' columns. If None, returns all fields.
        show_progress (bool): If True, show a progress bar, default False.

    Returns:
        `pandas.DataFrame`: DataFrame containing the observation metadata.
    """
    sequence_ids = listify(sequence_ids)

    observation_dfs = list()

    if show_progress:
        iterator = tqdm(sequence_ids)
    else:
        iterator = sequence_ids

    logger.debug(f'Getting images metadata for {len(sequence_ids)} files')
    for sequence_id in iterator:
        df_file = f'{OBS_BASE_URL}/{sequence_id}-metadata.parquet'
        if fields:
            fields = listify(fields)
            fields.insert(0, 'time')
            fields.insert(1, 'sequence_id')
            fields = list(set(fields))
        try:
            df = pd.read_parquet(df_file, columns=fields)
        except Exception as e:
            logger.warning(f'Problem reading {df_file}: {e!r}')
        else:
            observation_dfs.append(df)

    if len(observation_dfs) == 0:
        logger.debug(f'No documents found for sequence_ids={sequence_ids}')
        return

    df = pd.concat(observation_dfs)
    df = df.reindex(sorted(df.columns), axis=1)

    # TODO(wtgee) any data cleaning or preparation for observations here.

    return df.sort_values(by=['time'])


def search_observations(
        unit_id=None,
        start_date=None,
        end_date=None,
        ra=None,
        dec=None,
        coords=None,
        radius=10,  # degrees
        status=None,
        min_num_images=1,
        source_url='https://storage.googleapis.com/panoptes-exp.appspot.com/observations.csv',
        source=None
):
    """Search PANOPTES observations.

    >>> from astropy.coordinates import SkyCoord
    >>> from panoptes.utils.data import search_observations
    >>> coords = SkyCoord.from_name('Andromeda Galaxy')
    >>> start_date = '2019-01-01'
    >>> end_date = '2019-12-31'
    >>> search_results = search_observations(coords=coords, min_num_images=10, start_date=start_date, end_date=end_date)
    >>> # The result is a DataFrame you can further work with.
    >>> image_count = search_results.groupby(['unit_id', 'field_name']).num_images.sum()
    >>> image_count
    unit_id  field_name
    PAN001   Andromeda Galaxy     378
             HAT-P-19             148
             TESS_SEC17_CAM02    9949
    PAN012   Andromeda Galaxy      70
             HAT-P-16 b           268
             TESS_SEC17_CAM02    1983
    PAN018   TESS_SEC17_CAM02     244
    Name: num_images, dtype: Int64
    >>> print('Total minutes exposure:', search_results.total_minutes_exptime.sum())
    Total minutes exposure: 20376.83

    Args:
        ra (float|None): The RA position in degrees of the center of search.
        dec (float|None): The Dec position in degrees of the center of the search.
        coords (`astropy.coordinates.SkyCoord`|None): A valid coordinate instance.
        radius (float): The search radius in degrees. Searches are currently done in
            a square box, so this is half the length of the side of the box.
        start_date (str|`datetime.datetime`|None): A valid datetime instance or `None` (default).
            If `None` then the beginning of 2018 is used as a start date.
        end_date (str|`datetime.datetime`|None): A valid datetime instance or `None` (default).
            If `None` then today is used.
        unit_id (str|list|None): A str or list of strs of unit_ids to include.
            Default `None` will include all.
        status (str|list|None): A str or list of observation status to include.
            Default `None` will include all.
        min_num_images (int): Minimum number of images the observation should have, default 1.
        source_url (str): The remote url where the static CSV file is located.
        source (`pandas.DataFrame`|None): The dataframe to use or the search.
            If `None` (default) then the `source_url` will be used to look up the file.

    Returns:
        `pandas.DataFrame`: A table with the matching observation results.
    """

    logger.debug(f'Setting up search params')

    if coords is None:
        coords = SkyCoord(ra=ra, dec=dec, unit='degree')

    # Setup defaults for search.
    if start_date is None:
        start_date = '2018-01-01'

    if end_date is None:
        end_date = current_time()

    with suppress(TypeError):
        start_date = date_parse(start_date).replace(tzinfo=None)
    with suppress(TypeError):
        end_date = date_parse(end_date).replace(tzinfo=None)

    ra_max = (coords.ra + (radius * u.degree)).value
    ra_min = (coords.ra - (radius * u.degree)).value
    dec_max = (coords.dec + (radius * u.degree)).value
    dec_min = (coords.dec - (radius * u.degree)).value

    logger.debug(f'Getting list of observations')

    # Get the observation list
    obs_df = source
    if obs_df is None:
        local_path = download_file(source_url,
                                   cache='update',
                                   show_progress=False,
                                   pkgname='panoptes')
        obs_df = pd.read_csv(local_path).convert_dtypes()

    logger.debug(f'Found {len(obs_df)} total observations')

    # Perform filtering on other fields here.
    logger.debug(f'Filtering observations')
    obs_df.query(
        f'dec >= {dec_min} and dec <= {dec_max}'
        ' and '
        f'ra >= {ra_min} and ra <= {ra_max}'
        ' and '
        f'time >= "{start_date}"'
        ' and '
        f'time <= "{end_date}"'
        ' and '
        f'num_images >= {min_num_images}'
        ,
        inplace=True
    )
    logger.debug(f'Found {len(obs_df)} observations after initial filter')

    unit_ids = listify(unit_id)
    if len(unit_ids) > 0 and unit_ids != 'The Whole World! 🌎':
        obs_df.query(f'unit_id in {listify(unit_ids)}', inplace=True)
    logger.debug(f'Found {len(obs_df)} observations after unit filter')

    if status is not None:
        obs_df.query(f'status in {listify(status)}', inplace=True)
    logger.debug(f'Found {len(obs_df)} observations after status filter')

    logger.debug(f'Found {len(obs_df)} observations after filtering')

    obs_df = obs_df.reindex(sorted(obs_df.columns), axis=1)
    obs_df.sort_values(by=['time'], inplace=True)

    # TODO(wtgee) any data cleaning or preparation for observations here.

    columns = [
        'sequence_id',
        'unit_id',
        'camera_id',
        'ra',
        'dec',
        'exptime',
        'field_name',
        'iso',
        'num_images',
        'software_version',
        'status',
        'time',
        'total_minutes_exptime',
    ]

    return obs_df.reindex(columns=columns)
