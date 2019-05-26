import os
import shutil
import subprocess
from contextlib import suppress

import pandas as pd
from tqdm import tqdm

from collections import namedtuple

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time
from astropy.stats import sigma_clipped_stats

from panoptes.utils.google.cloudsql import get_cursor
from panoptes.utils.images import fits as fits_utils
from panoptes.utils.logger import get_root_logger

from panoptes.piaa.utils import helpers

import logging
logger = get_root_logger()
logger.setLevel(logging.DEBUG)


def get_stars_from_footprint(wcs_footprint, **kwargs):
    """Lookup star information from WCS footprint.

    This is just a thin wrapper around `get_stars`.

    Args:
        wcs_footprint (`astropy.wcs.WCS`): The world coordinate system (WCS) for an image.
        **kwargs: Optional keywords to pass to `get_stars`.

    Returns:
        TYPE: Description
    """
    ra = wcs_footprint[:, 0]
    dec = wcs_footprint[:, 1]

    logger.debug(f'WCS footprint: {ra} {dec}')

    return get_stars(ra.min(), ra.max(), dec.min(), dec.max(), **kwargs)


def get_stars(
        ra_min,
        ra_max,
        dec_min,
        dec_max,
        vmag=13,
        table='full_catalog',
        cursor=None,
        cursor_only=True,
        verbose=False,
        **kwargs):
    """Look star information from the TESS catalog.

    Args:
        ra_min (float): The minimum RA in degrees.
        ra_max (float): The maximum RA in degrees.
        dec_min (float): The minimum Dec in degress.
        dec_max (float): The maximum Dec in degrees.
        table (str, optional): TESS catalog table to use, default 'full_catalog'. Can also be 'ctl'
        cursor_only (bool, optional): Return raw cursor, default False.
        verbose (bool, optional): Verbose, default False.
        **kwargs: Additional keyword arrs passed to `get_cursor`.

    Returns:
        `astropy.table.Table` or `psycopg2.cursor`: Table with star information be default,
            otherwise the raw cursor if `cursor_only=True`.
    """
    if not cursor:
        logger.info(f'No DB cursor, getting new one')
        cursor = get_cursor(
            instance='panoptes-tess-catalog',
            db_name='v702',
            db_user='postgres',
            port=5433,
            **kwargs)

    logger.debug(f'About to execute SELECT sql')
    logger.debug('{} {} {} {}'.format(ra_min, ra_max, dec_min, dec_max))
    cursor.execute("""SELECT id, ra, dec, tmag, vmag, e_tmag, twomass
        FROM {}
        WHERE ra >= %s AND ra <= %s AND dec >= %s AND dec <= %s AND vmag < %s;""".format(table),
                   (ra_min, ra_max, dec_min, dec_max, vmag)
                   )
    if cursor_only:
        logger.debug(f'Returning cursor')
        return cursor

    logger.debug('Building table of results')
    d0 = cursor.fetchall()
    logger.debug(f'Fetched {len(d0)} sources')
    try:
        source_table = Table(
            data=d0,
            names=['id', 'ra', 'dec', 'tmag', 'vmag', 'e_tmag', 'twomass'],
            dtype=['i4', 'f8', 'f8', 'f4', 'f4', 'f4', 'U26'])
    except TypeError:
        logger.warn(f'Error building star catalog table')
        return None
    else:
        logger.debug(f'Returning source table')
        return source_table


def get_star_info(picid=None,
                  twomass_id=None,
                  table='full_catalog',
                  cursor=None,
                  raw=False,
                  verbose=False,
                  **kwargs):
    """Lookup catalog information about a given star.

    Args:
        picid (str, optional): The ID of the star in the TESS catalog, also
            known as the PANOPTES Input Catalog ID (picid).
        twomass_id (str, optional): 2Mass ID.
        table (str, optional): TESS catalog table to use, default 'full_catalog'.
            Can also be 'ctl'.
        verbose (bool, optional): Verbose, default False.
        **kwargs: Description

    Returns:
        tuple: Values from the database.
    """
    if not cursor:
        cursor = get_cursor(db_name='v6', db_user='postgres', **kwargs)

    if picid:
        val = picid
        col = 'id'
    elif twomass_id:
        val = twomass_id
        col = 'twomass'

    cursor.execute('SELECT * FROM {} WHERE {}=%s'.format(table, col), (val,))

    rec = cursor.fetchone()

    if rec and not raw:
        StarInfo = namedtuple('StarInfo', sorted(rec.keys()))
        rec = StarInfo(**rec)

    return rec


def lookup_sources_for_observation(fits_files=None,
                                   filename=None,
                                   force_new=False,
                                   cursor=None,
                                   use_intersection=False,
                                   **kwargs
                                   ):

    if force_new:
        logger.info(f'Forcing a new source file')
        with suppress(FileNotFoundError):
            os.remove(filename)

    try:
        logger.info(f'Using existing source file: {filename}')
        observation_sources = pd.read_csv(filename, parse_dates=True)
        observation_sources['obstime'] = pd.to_datetime(observation_sources.obstime)

    except FileNotFoundError:
        if not cursor:
            cursor = get_cursor(port=5433, db_name='v702', db_user='panoptes')

        logger.info(f'Looking up sources in {len(fits_files)} files')
        observation_sources = None

        # Lookup the point sources for all frames
        for fn in tqdm(fits_files):
            point_sources = lookup_point_sources(
                fn,
                force_new=force_new,
                cursor=cursor,
                **kwargs
            )
            header = fits_utils.getheader(fn)
            obstime = Time(pd.to_datetime(os.path.basename(fn).split('.')[0]))
            exptime = header['EXPTIME'] * u.second

            obstime += (exptime / 2)

            point_sources['obstime'] = obstime.datetime
            point_sources['exptime'] = exptime
            point_sources['airmass'] = header['AIRMASS']
            point_sources['file'] = os.path.basename(fn)
            point_sources['picid'] = point_sources.index

            logger.info(f'Combining sources with previous observations')
            if observation_sources is not None:
                if use_intersection:
                    logger.info(f'Getting intersection of sources')

                    idx_intersection = observation_sources.index.intersection(point_sources.index)
                    logger.info(f'Num sources in intersection: {len(idx_intersection)}')
                    observation_sources = pd.concat([observation_sources.loc[idx_intersection],
                                                     point_sources.loc[idx_intersection]],
                                                    join='inner')
                else:
                    observation_sources = pd.concat([observation_sources, point_sources])
            else:
                observation_sources = point_sources

        logger.info(f'Writing sources out to file')
        observation_sources.to_csv(filename)

    observation_sources.set_index(['obstime'], inplace=True)
    return observation_sources


def lookup_point_sources(fits_file,
                         catalog_match=True,
                         method='sextractor',
                         force_new=False,
                         **kwargs
                         ):
    """ Extract point sources from image

    Args:
        fits_file (str, optional): Path to FITS file to search for stars.
        force_new (bool, optional): Force a new catalog to be created,
            defaults to False

    Raises:
        error.InvalidSystemCommand: Description
    """
    if catalog_match or method == 'tess_catalog':
        fits_header = fits_utils.getheader(fits_file)
        wcs = WCS(fits_header)
        assert wcs is not None and wcs.is_celestial, logger.warning("Need a valid WCS")

    logger.info("Looking up sources for {}".format(fits_file))

    lookup_function = {
        'sextractor': _lookup_via_sextractor,
        'tess_catalog': _lookup_via_tess_catalog,
        'photutils': _lookup_via_photutils,
    }

    # Lookup our appropriate method and call it with the fits file and kwargs
    try:
        logger.debug("Using {} method {}".format(method, lookup_function[method]))
        point_sources = lookup_function[method](fits_file, force_new=force_new, **kwargs)
    except Exception as e:
        logger.error("Problem looking up sources: {}".format(e))
        raise Exception("Problem lookup up sources: {}".format(e))

    if catalog_match:
        logger.debug(f'Doing catalog match against stars')
        point_sources = get_catalog_match(point_sources, wcs, **kwargs)
        logger.debug(f'Done with catalog match')

    # Change the index to the picid
    point_sources.set_index('id', inplace=True)
    point_sources.index.rename('picid', inplace=True)

    # Remove those with more than one entry
    counts = point_sources.x.groupby('picid').count()
    single_entry = counts == 1
    single_index = single_entry.loc[single_entry].index
    unique_sources = point_sources.loc[single_index]

    return unique_sources


def get_catalog_match(point_sources, wcs, table='full_catalog', **kwargs):
    assert point_sources is not None

    # Get coords from detected point sources
    stars_coords = SkyCoord(
        ra=point_sources['ra'].values * u.deg,
        dec=point_sources['dec'].values * u.deg
    )

    # Lookup stars in catalog
    logger.debug(f'Getting catalog stars')
    catalog_stars = helpers.get_stars_from_footprint(
        wcs.calc_footprint(),
        cursor_only=False,
        table=table,
        **kwargs
    )
    if catalog_stars is None:
        logger.warn('No catalog matches, returning table without ids')
        return point_sources

    logger.debug(f'Matched {len(catalog_stars)} sources to catalog')

    # Get coords for catalog stars
    catalog_coords = SkyCoord(
        ra=catalog_stars['ra'] * u.deg,
        dec=catalog_stars['dec'] * u.deg
    )

    # Do catalog matching
    logger.debug(f'Doing actual match')
    idx, d2d, d3d = match_coordinates_sky(stars_coords, catalog_coords)
    logger.debug(f'Got matched sources')

    # Get some properties from the catalog
    point_sources['id'] = catalog_stars[idx]['id']
    # point_sources['twomass'] = catalog_stars[idx]['twomass']
    point_sources['tmag'] = catalog_stars[idx]['tmag']
    point_sources['vmag'] = catalog_stars[idx]['vmag']
    point_sources['catalog_sep_arcsec'] = d2d.to(u.arcsec).value

    return point_sources


def _lookup_via_sextractor(fits_file, sextractor_params=None, *args, **kwargs):
    # Write the sextractor catalog to a file
    base_dir = os.path.dirname(fits_file)
    source_dir = os.path.join(base_dir, 'sextractor')
    os.makedirs(source_dir, exist_ok=True)

    img_id = os.path.splitext(os.path.basename(fits_file))[0]

    source_file = os.path.join(source_dir, f'point_sources_{img_id}.cat')

    # sextractor can't handle compressed data
    if fits_file.endswith('.fz'):
        fits_file = fits_utils.funpack(fits_file)

    logger.info("Point source catalog: {}".format(source_file))

    if not os.path.exists(source_file) or kwargs.get('force_new', False):
        logger.info("No catalog found, building from sextractor")
        # Build catalog of point sources
        sextractor = shutil.which('sextractor')
        if sextractor is None:
            sextractor = shutil.which('sex')
            if sextractor is None:
                raise Exception('sextractor not found')

        if sextractor_params is None:
            sextractor_params = [
                '-c', '{}/PIAA/resources/conf_files/sextractor/panoptes.sex'.format(
                    os.getenv('PANDIR')),
                '-CATALOG_NAME', source_file,
            ]

        logger.info("Running sextractor...")
        cmd = [sextractor, *sextractor_params, fits_file]
        logger.info(cmd)

        try:
            subprocess.run(cmd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           timeout=60,
                           check=True)
        except subprocess.CalledProcessError as e:
            raise Exception("Problem running sextractor: {}".format(e))

    # Read catalog
    logger.info('Building detected source table')
    point_sources = Table.read(source_file, format='ascii.sextractor')

    # Remove the point sources that sextractor has flagged
    # if 'FLAGS' in point_sources.keys():
    #    point_sources = point_sources[point_sources['FLAGS'] == 0]
    #    point_sources.remove_columns(['FLAGS'])

    # Rename columns
    point_sources.rename_column('XPEAK_IMAGE', 'x')
    point_sources.rename_column('YPEAK_IMAGE', 'y')

    # Filter point sources near edge
    # w, h = data[0].shape
    w, h = (3476, 5208)

    stamp_size = 60

    logger.info('Trimming sources near edge')
    top = point_sources['y'] > stamp_size
    bottom = point_sources['y'] < w - stamp_size
    left = point_sources['x'] > stamp_size
    right = point_sources['x'] < h - stamp_size

    point_sources = point_sources[top & bottom & right & left].to_pandas()
    point_sources.columns = [
        'ra', 'dec',
        'x', 'y',
        'x_image', 'y_image',
        'background',
        'flux_best', 'fluxerr_best',
        'mag_best', 'magerr_best',
        'flux_max',
        'fwhm_image',
        'flags',
    ]

    logger.info(f'Returning {len(point_sources)} sources')
    return point_sources


def _lookup_via_tess_catalog(fits_file, wcs=None, *args, **kwargs):
    wcs_footprint = wcs.calc_footprint()
    logger.info("WCS footprint: {}".format(wcs_footprint))

    # Get stars from TESS catalog
    point_sources = helpers.get_stars_from_footprint(
        wcs_footprint,
        cursor_only=False,
        table=kwargs.get('table', 'full_catalog')
    )

    # Get x,y coordinates
    star_pixels = wcs.all_world2pix(point_sources['ra'], point_sources['dec'], 0)
    point_sources['x'] = star_pixels[0]
    point_sources['y'] = star_pixels[1]

    point_sources.add_index(['id'])
    point_sources = point_sources.to_pandas()

    return point_sources


def _lookup_via_photutils(fits_file, wcs=None, *args, **kwargs):
    from photutils import DAOStarFinder
    data = fits.getdata(fits_file) - 2048  # Camera bias
    mean, median, std = sigma_clipped_stats(data)

    fwhm = kwargs.get('fwhm', 3.0)
    threshold = kwargs.get('threshold', 3.0)

    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources = daofind(data - median).to_pandas()

    sources.rename(columns={
        'xcentroid': 'x',
        'ycentroid': 'y',
    }, inplace=True)

    if wcs is None:
        header = fits_utils.getheader(fits_file)
        wcs = WCS(header)

    coords = wcs.all_pix2world(sources['x'], sources['y'], 1)

    sources['ra'] = coords[0]
    sources['dec'] = coords[1]

    return sources
