import os
import shutil
import subprocess
from contextlib import suppress

import pandas as pd
from tqdm import tqdm

from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time

from panoptes.utils.images import fits as fits_utils
from panoptes.utils.google.cloudsql import get_cursor
from panoptes.piaa.utils import helpers

import logging
logger = logging.getLogger(__name__)


def lookup_sources_for_observation(fits_files=None,
                                   filename=None,
                                   force_new=False,
                                   cursor=None,
                                   use_intersection=False,
                                   **kwargs
                                   ):

    if force_new:
        print(f'Forcing a new source file')
        with suppress(FileNotFoundError):
            os.remove(filename)

    try:
        print(f'Using existing source file: {filename}')
        observation_sources = pd.read_csv(filename, parse_dates=True)
        observation_sources['obstime'] = pd.to_datetime(observation_sources.obstime)

    except FileNotFoundError:
        if not cursor:
            cursor = get_cursor(port=5433, db_name='v702', db_user='panoptes')

        print(f'Looking up sources in {len(fits_files)} files')
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

            print(f'Combining sources with previous observations')
            if observation_sources is not None:
                if use_intersection:
                    print(f'Getting intersection of sources')

                    idx_intersection = observation_sources.index.intersection(point_sources.index)
                    print(f'Num sources in intersection: {len(idx_intersection)}')
                    observation_sources = pd.concat([observation_sources.loc[idx_intersection],
                                                     point_sources.loc[idx_intersection]],
                                                    join='inner')
                else:
                    observation_sources = pd.concat([observation_sources, point_sources])
            else:
                observation_sources = point_sources

        print(f'Writing sources out to file')
        observation_sources.to_csv(filename)

    observation_sources.set_index(['obstime'], inplace=True)
    return observation_sources


def lookup_point_sources(fits_file,
                         catalog_match=True,
                         method='sextractor',
                         force_new=False,
                         max_catalog_separation=25,  # arcsecs
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
    def _print(msg):
        if 'logger' in kwargs:
            logger.debug(msg)
        else:
            print(msg)

    if catalog_match or method == 'tess_catalog':
        fits_header = fits_utils.getheader(fits_file)
        wcs = WCS(fits_header)
        assert wcs is not None and wcs.is_celestial, logging.warning("Need a valid WCS")

    _print(f"Looking up sources for {fits_file}")

    lookup_function = {
        'sextractor': _lookup_via_sextractor,
        'tess_catalog': _lookup_via_tess_catalog,
    }

    # Lookup our appropriate method and call it with the fits file and kwargs
    try:
        _print(f"Using {method} method for {fits_file}")
        point_sources = lookup_function[method](fits_file, force_new=force_new, **kwargs)
    except Exception as e:
        _print(f"Problem looking up sources: {e!r} {fits_file}")
        raise Exception(f"Problem looking up sources: {e!r} {fits_file}")

    if catalog_match:
        _print(f'Doing catalog match against stars {fits_file}')
        try:
            point_sources = get_catalog_match(point_sources, wcs, **kwargs)
        except Exception as e:
            _print(f'Error in catalog match: {e!r} {fits_file}')
        _print(f'Done with catalog match {fits_file}')

    # Change the index to the picid
    point_sources.set_index('picid', inplace=True)
    _print(f'Point sources: {len(point_sources)}')

    # Remove catalog matches that are too large
    _print(f'Removing matches that are greater than {max_catalog_separation} arcsec from catalog.')
    point_sources = point_sources.loc[point_sources.catalog_sep_arcsec < max_catalog_separation]
    _print(f'Point sources: {len(point_sources)} {fits_file}')

    return point_sources


def get_catalog_match(point_sources, wcs, table='full_catalog', **kwargs):
    assert point_sources is not None

    def _print(msg):
        if 'logger' in kwargs:
            logger.debug(msg)
        else:
            print(msg)

    _print(f'Getting catalog stars')

    # Get coords from detected point sources
    stars_coords = SkyCoord(
        ra=point_sources['ra'].values * u.deg,
        dec=point_sources['dec'].values * u.deg
    )

    # Lookup stars in catalog
    catalog_stars = helpers.get_stars_from_footprint(
        wcs.calc_footprint(),
        cursor_only=False,
        table=table,
        **kwargs
    )
    if catalog_stars is None:
        _print('No catalog matches, returning table without ids')
        return point_sources

    _print(f'Found {len(catalog_stars)} catalog sources in WCS footprint')

    # Get coords for catalog stars
    catalog_coords = SkyCoord(
        ra=catalog_stars['ra'] * u.deg,
        dec=catalog_stars['dec'] * u.deg
    )

    # Do catalog matching
    _print(f'Matching catalog')
    idx, d2d, d3d = match_coordinates_sky(stars_coords, catalog_coords)
    _print(f'Got {len(idx)} matched sources (includes duplicates)')

    # _print(f'Adding catalog_stars columns: {catalog_stars.columns}')

    # Get some properties from the catalog
    point_sources['picid'] = catalog_stars.iloc[idx]['id'].values
    # point_sources['twomass'] = catalog_stars[idx]['twomass']
    point_sources['tmag'] = catalog_stars.iloc[idx]['tmag'].values
    point_sources['tmag_err'] = catalog_stars.iloc[idx]['e_tmag'].values
    point_sources['vmag'] = catalog_stars.iloc[idx]['vmag'].values
    point_sources['vmag_err'] = catalog_stars.iloc[idx]['e_vmag'].values
    point_sources['lumclass'] = catalog_stars.iloc[idx]['lumclass'].values
    point_sources['lum'] = catalog_stars.iloc[idx]['lum'].values
    point_sources['lum_err'] = catalog_stars.iloc[idx]['e_lum'].values
    # Contamination ratio
    point_sources['contratio'] = catalog_stars.iloc[idx]['contratio'].values
    # Number of sources in TESS aperture
    point_sources['numcont'] = catalog_stars.iloc[idx]['numcont'].values
    point_sources['catalog_sep_arcsec'] = d2d.to(u.arcsec).value

    # _print(f'point_sources.columns: {point_sources.columns}')

    return point_sources


def _lookup_via_sextractor(fits_file, sextractor_params=None, *args, **kwargs):

    def _print(msg):
        if 'logger' in kwargs:
            logger.debug(msg)
        else:
            print(msg)

    # Write the sextractor catalog to a file
    base_dir = os.path.dirname(fits_file)
    source_dir = os.path.join(base_dir, 'sextractor')
    os.makedirs(source_dir, exist_ok=True)

    img_id = os.path.splitext(os.path.basename(fits_file))[0]

    source_file = os.path.join(source_dir, f'point_sources_{img_id}.cat')

    # sextractor can't handle compressed data
    if fits_file.endswith('.fz'):
        fits_file = fits_utils.funpack(fits_file)

    _print("Point source catalog: {}".format(source_file))

    if not os.path.exists(source_file) or kwargs.get('force_new', False):
        _print("No catalog found, building from sextractor")
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

        _print("Running sextractor...")
        cmd = [sextractor, *sextractor_params, fits_file]
        _print(cmd)

        try:
            subprocess.run(cmd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           timeout=60,
                           check=True)
        except subprocess.CalledProcessError as e:
            raise Exception("Problem running sextractor: {}".format(e))

    # Read catalog
    _print('Building detected source table')
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

    _print('Trimming sources near edge')
    top = point_sources['y'] > stamp_size
    bottom = point_sources['y'] < w - stamp_size
    left = point_sources['x'] > stamp_size
    right = point_sources['x'] < h - stamp_size

    point_sources = point_sources[top & bottom & right & left].to_pandas()
    point_sources.columns = [
        'ra', 'dec',
        'x', 'y',
        'x_image', 'y_image',
        'flux_best', 'fluxerr_best',
        'mag_best', 'magerr_best',
        'flux_max',
        'fwhm_image',
        'flags',
    ]

    _print(f'Returning {len(point_sources)} sources from sextractor')
    return point_sources


def _lookup_via_tess_catalog(fits_file, wcs=None, *args, **kwargs):
    wcs_footprint = wcs.calc_footprint()

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
