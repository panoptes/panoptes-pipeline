import re
import traceback
from enum import IntEnum, auto
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Pattern, Union, Optional, Tuple

from dateutil.parser import parse as parse_date
from dateutil.tz import UTC
import pandas as pd
from google.cloud import firestore
from tqdm.auto import tqdm
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io.fits.header import Header
from astropy.utils.data import download_file

from loguru import logger

from panoptes.utils.utils import listify
from panoptes.utils.time import current_time, flatten_time
from panoptes.utils.images import fits as fits_utils


class SequenceStatus(IntEnum):
    RECEIVING = 0
    RECEIVED = 10


class ImageStatus(IntEnum):
    ERROR = -10
    MASKED = -5
    UNKNOWN = -1
    RECEIVING = 0
    RECEIVED = 5
    CALIBRATING = 10
    CALIBRATED = 15
    SOLVING = 20
    SOLVED = 25
    MATCHING = 30
    MATCHED = 35
    EXTRACTING = 40
    EXTRACTED = 45


class ObservationStatus(IntEnum):
    ERROR = -10
    UNKNOWN = -1
    CREATED = 0
    CALIBRATED = 10
    MATCHED = 20
    EXTRACTED = 30
    PROCESSED = 40


OBS_BASE_URL = 'https://storage.googleapis.com/panoptes-observations'
OBSERVATIONS_URL = 'https://storage.googleapis.com/panoptes-exp.appspot.com/observations.csv'

PATH_MATCHER: Pattern[str] = re.compile(r"""^
                                (?P<pre_info>.*)?                       # Anything before unit_id
                                (?P<unit_id>PAN\d{3})                   # unit_id   - PAN + 3 digits
                                /(?P<camera_id>[a-gA-G0-9]{6})          # camera_id - 6 digits
                                /?(?P<field_name>.*)?                   # Legacy field name - any
                                /(?P<sequence_time>[0-9]{8}T[0-9]{6})   # Observation start time
                                /(?P<image_time>[0-9]{8}T[0-9]{6})      # Image start time
                                (?P<post_info>.*)?                      # Anything after (file ext)
                                $""",
                                        re.VERBOSE)


@dataclass
class ObservationPathInfo:
    """Parse the location path for an image.

    This is a small dataclass that offers some convenience methods for dealing
    with a path based on the image id.

    This would usually be instantiated via `path`:

    ..doctest::

        >>> from panoptes.pipeline.utils.metadata import ObservationPathInfo
        >>> bucket_path = 'gs://panoptes-images-background/PAN012/Hd189733/358d0f/20180824T035917/20180824T040118.fits'
        >>> path_info = ObservationPathInfo(path=bucket_path)

        >>> path_info.id
        'PAN012_358d0f_20180824T035917_20180824T040118'

        >>> path_info.unit_id
        'PAN012'

        >>> path_info.sequence_id
        'PAN012_358d0f_20180824T035917'

        >>> path_info.image_id
        'PAN012_358d0f_20180824T040118'

        >>> path_info.as_path(base='/tmp', ext='.jpg')
        '/tmp/PAN012/358d0f/20180824T035917/20180824T040118.jpg'

        >>> ObservationPathInfo(path='foobar')
        Traceback (most recent call last):
          ...
        ValueError: Invalid path received: self.path='foobar'


    """
    unit_id: str = None
    camera_id: str = None
    field_name: str = None
    sequence_time: Union[str, datetime, Time] = None
    image_time: Union[str, datetime, Time] = None
    path: Union[str, Path] = None

    def __post_init__(self):
        """Parse the path when provided upon initialization."""
        if self.path is not None:
            path_match = PATH_MATCHER.match(self.path)
            if path_match is None:
                raise ValueError(f'Invalid path received: {self.path=}')

            self.unit_id = path_match.group('unit_id')
            self.camera_id = path_match.group('camera_id')
            self.field_name = path_match.group('field_name')
            self.sequence_time = Time(parse_date(path_match.group('sequence_time')))
            self.image_time = Time(parse_date(path_match.group('image_time')))

    @property
    def id(self):
        """Full path info joined with underscores"""
        return self.get_full_id()

    @property
    def sequence_id(self) -> str:
        """The sequence id."""
        return f'{self.unit_id}_{self.camera_id}_{flatten_time(self.sequence_time)}'

    @property
    def image_id(self) -> str:
        """The matched image id."""
        return f'{self.unit_id}_{self.camera_id}_{flatten_time(self.image_time)}'

    def as_path(self, base: Union[Path, str] = None, ext: str = None) -> Path:
        """Return a Path object."""
        image_str = flatten_time(self.image_time)
        if ext is not None:
            image_str = f'{image_str}.{ext}'

        full_path = Path(self.unit_id, self.camera_id, flatten_time(self.sequence_time), image_str)

        if base is not None:
            full_path = base / full_path

        return full_path

    def get_full_id(self, sep='_') -> str:
        """Returns the full path id with the given separator."""
        return f'{sep}'.join([
            self.unit_id,
            self.camera_id,
            flatten_time(self.sequence_time),
            flatten_time(self.image_time)
        ])

    @classmethod
    def from_fits(cls, fits_file):
        header = fits_utils.getheader(fits_file)
        return cls.from_fits_header(header)

    @classmethod
    def from_fits_header(cls, header):
        try:
            new_instance = cls(path=header['FILENAME'])
        except ValueError:
            sequence_id = header['SEQID']
            image_id = header['IMAGEID']
            unit_id, camera_id, sequence_time = sequence_id.split('_')
            _, _, image_time = image_id.split('_')

            new_instance = cls(unit_id=unit_id,
                               camera_id=camera_id,
                               sequence_time=Time(parse_date(sequence_time)),
                               image_time=Time(parse_date(image_time)))

        return new_instance


def extract_metadata(header: Header) -> dict:
    """Get the metadata from a FITS image."""
    path_info = ObservationPathInfo.from_fits_header(header)

    try:
        measured_rggb = [float(x) for x in header.get('MEASRGGB', '0 0 0 0').split(' ')]
        if 'DATE' in header:
            file_date = parse_date(header.get('DATE')).replace(tzinfo=UTC)
        else:
            file_date = path_info.image_time.to_datetime(timezone=UTC)
        camera_date = parse_date(header.get('DATE-OBS', path_info.image_time)).replace(tzinfo=UTC)

        unit_info = dict(
            unit_id=path_info.unit_id,
            latitude=header.get('LAT-OBS'),
            longitude=header.get('LONG-OBS'),
            elevation=float(header.get('ELEV-OBS'))
        )

        sequence_info = dict(
            unit_id=path_info.unit_id,
            sequence_id=path_info.sequence_id,
            time=path_info.sequence_time.to_datetime(timezone=UTC),
            coordinates=dict(
                mount_dec=header.get('DEC-MNT'),
                mount_ra=header.get('RA-MNT'),
            ),
            camera=dict(
                camera_id=path_info.camera_id,
                lens_serial_number=header.get('INTSN'),
                serial_number=str(header.get('CAMSN')),
            ),
            field_name=header.get('FIELD', ''),
            software_version=header.get('CREATOR', ''),
        )

        image_info = dict(
            uid=path_info.get_full_id(sep='_'),
            camera=dict(
                dateobs=camera_date,
                exptime=float(header.get('EXPTIME')),
                iso=header.get('ISO'),
                circconf=float(header.get('CIRCCONF', '0.').split(' ')[0]),
                blue_balance=float(header.get('BLUEBAL')),
                red_balance=float(header.get('REDBAL')),
                colortemp=float(header.get('COLORTMP')),
                measured_rggb=measured_rggb,
                measured_ev=[float(header.get('MEASEV')), float(header.get('MEASEV2'))],
                temperature=float(header.get('CAMTEMP', 0).split(' ')[0]),
                white_lvln=header.get('WHTLVLN'),
                white_lvls=header.get('WHTLVLS'),
            ),
            environment=dict(
                moonfrac=float(header.get('MOONFRAC')),
                moonsep=float(header.get('MOONSEP')),
            ),
            coordinates=dict(
                airmass=header.get('AIRMASS'),
                mount_ha=header.get('HA-MNT'),
                ra=header.get('CRVAL1'),
                dec=header.get('CRVAL2'),
            ),
            file_creation_date=file_date,
            time=path_info.image_time.to_datetime(timezone=UTC),
        )

    except Exception as e:
        logger.error(f'Error in adding record: {e!r}')
        raise e

    logger.success(f'Metadata extracted from header')
    return dict(unit=unit_info, sequence=sequence_info, image=image_info)


def get_observation_metadata(sequence_ids, fields=None, show_progress=False):
    """Get the metadata for the given sequence_id(s).

    NOTE: This is slated for removal soon.

    This function will search for pre-processed observations that have a stored
    parquet file.

    Note that since the files are stored in parquet format, specifying the `fields`
    does in fact save on the size of the returned data. If requesting many `sequence_ids`
    it may be worth figuring out exactly what columns you need first.

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
        iterator = tqdm(sequence_ids, desc='Getting image metadata')
    else:
        iterator = sequence_ids

    logger.debug(f'Getting images metadata for {len(sequence_ids)} files')
    for sequence_id in iterator:
        df_file = f'{OBS_BASE_URL}/{sequence_id}-metadata.parquet'
        if fields:
            fields = listify(fields)
            # Always return the ID fields.
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
        logger.info(f'No documents found for sequence_ids={sequence_ids}')
        return

    df = pd.concat(observation_dfs)

    # Return column names in sorted order
    df = df.reindex(sorted(df.columns), axis=1)

    # TODO(wtgee) any data cleaning or preparation for observations here.

    logger.success(f'Returning {len(df)} rows of metadata sorted by time')
    return df.sort_values(by=['time'])


def search_observations(
        coords=None,
        unit_id=None,
        start_date=None,
        end_date=None,
        ra=None,
        dec=None,
        radius=10,  # degrees
        status='matched',
        min_num_images=1,
        source_url=OBSERVATIONS_URL,
        source=None
):
    """Search PANOPTES observations.

    Either a `coords` or `ra` and `dec` must be specified for search to work.

    >>> from astropy.coordinates import SkyCoord
    >>> from panoptes.pipeline.utils.metadata import search_observations
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
        coords (`astropy.coordinates.SkyCoord`|None): A valid coordinate instance.
        ra (float|None): The RA position in degrees of the center of search.
        dec (float|None): The Dec position in degrees of the center of the search.
        radius (float): The search radius in degrees. Searches are currently done in
            a square box, so this is half the length of the side of the box.
        start_date (str|`datetime.datetime`|None): A valid datetime instance or `None` (default).
            If `None` then the beginning of 2018 is used as a start date.
        end_date (str|`datetime.datetime`|None): A valid datetime instance or `None` (default).
            If `None` then today is used.
        unit_id (str|list|None): A str or list of strs of unit_ids to include.
            Default `None` will include all.
        status (str|list|None): A str or list of observation status to include.
            Defaults to "matched" for observations that have been fully processed. Passing
            `None` will return all status.
        min_num_images (int): Minimum number of images the observation should have, default 1.
        source_url (str): The remote url where the static CSV file is located, default to PANOPTES
            storage location.
        source (`pandas.DataFrame`|None): The dataframe to use or the search.
            If `None` (default) then the `source_url` will be used to look up the file.

    Returns:
        `pandas.DataFrame`: A table with the matching observation results.
    """

    logger.debug(f'Setting up search params')

    if coords is None:
        try:
            coords = SkyCoord(ra=ra, dec=dec, unit='degree')
        except ValueError:
            raise

            # Setup defaults for search.
    if start_date is None:
        start_date = '2018-01-01'

    if end_date is None:
        end_date = current_time()

    with suppress(TypeError):
        start_date = parse_date(start_date).replace(tzinfo=None)
    with suppress(TypeError):
        end_date = parse_date(end_date).replace(tzinfo=None)

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
        obs_df = pd.read_csv(local_path)

    logger.info(f'Found {len(obs_df)} total observations')

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

    logger.success(f'Returning {len(obs_df)} observations')
    return obs_df.reindex(columns=columns)


def get_firestore_doc_ref(
        bucket_path: str,
        unit_collection: str = 'units',
        observation_collection: str = 'observations',
        image_collection: str = 'images',
        firestore_db: firestore.Client = None
) -> Tuple[firestore.DocumentReference, firestore.DocumentReference, firestore.DocumentReference]:
    """Gets the firestore image document reference"""
    firestore_db = firestore_db or firestore.Client()

    path_info = ObservationPathInfo(path=bucket_path)
    sequence_id = path_info.sequence_id
    image_id = path_info.image_id

    print(f'Getting document for image {path_info.get_full_id()}')
    unit_collection_ref = firestore_db.collection(unit_collection)
    unit_doc_ref = unit_collection_ref.document(f'{path_info.unit_id}')
    seq_doc_ref = unit_doc_ref.collection(observation_collection).document(sequence_id)
    image_doc_ref = seq_doc_ref.collection(image_collection).document(image_id)

    return unit_doc_ref, seq_doc_ref, image_doc_ref


def record_metadata(bucket_path: str, metadata: dict, **kwargs) -> str:
    """Add FITS header info to firestore_db.

    Note:
        This function doesn't check header for proper entries and
        assumes a large list of keywords. See source for details.

    Returns:
        str: The image_id.

    Raises:
        e: Description
    """
    # TODO support batch operation.
    if not metadata:
        raise RuntimeError('Need valid metadata')

    print(f'Recording header metadata in firestore for {bucket_path=}')

    path_info = ObservationPathInfo(path=bucket_path)
    sequence_id = path_info.sequence_id

    try:
        print(f'Getting document for observation {sequence_id}')
        unit_doc_ref, sequence_doc_ref, image_doc_ref = get_firestore_doc_ref(bucket_path, **kwargs)

        metadata['unit']['num_images'] = firestore.Increment(1)
        metadata['sequence']['num_images'] = firestore.Increment(1)

        # Increment exptime totals.
        with suppress(KeyError):
            exptime = float(metadata['image']['camera']['exptime'])
            metadata['unit']['total_exptime'] = firestore.Increment(exptime)
            metadata['sequence']['total_exptime'] = firestore.Increment(exptime)

        unit_doc_ref.set(metadata['unit'], merge=True)
        sequence_doc_ref.set(metadata['sequence'], merge=True)

        metadata['image']['received_time'] = firestore.SERVER_TIMESTAMP
        image_doc_ref.set(metadata['image'], merge=True)

    except Exception as e:
        print(f'Error in adding record: {traceback.format_exc()!r}')
        raise e
    else:
        print(f'Recorded metadata for {path_info.get_full_id()} with {image_doc_ref.id=}')
        return image_doc_ref.path
