import os
import shutil
import subprocess

import pkg_resources
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
from panoptes.pipeline.utils import get_project_root

from panoptes.utils.images import fits as fits_utils
from panoptes.utils.logging import logger

from panoptes.pipeline.utils.gcp.bigquery import get_bq_clients


def get_stars_from_wcs(wcs, **kwargs):
    """Lookup star information from WCS footprint.

    Generates the correct layout for an SQL `POLYGON` that can be passed to
    :py:func:`get_stars`.

    Args:
        wcs (`astropy.wcs.WCS` or array): A valid (i.e. `wcs.is_celestial`) World Coordinate System object.
        **kwargs: Optional keywords to pass to :py:func:`get_stars`.

    """
    wcs_footprint = wcs.calc_footprint().tolist()
    logger.debug(f'Looking up catalog stars for WCS: {wcs_footprint}')
    # Add the first entry to the end to complete polygon
    wcs_footprint.append(wcs_footprint[0])

    poly = ','.join([f'{c[0]:.05} {c[1]:.05f}' for c in wcs_footprint])

    logger.debug(f'Using poly={poly} for get_stars')
    catalog_stars = get_stars(shape=poly, **kwargs)

    return catalog_stars


def get_stars(
        shape=None,
        vmag_min=4,
        vmag_max=17,
        bq_client=None,
        bqstorage_client=None,
        return_dataframe=True,
        **kwargs):
    """Look star information from the TESS catalog.

    Args:
        shape (str, optional): A string representation of an SQL shape, e.g. `POLYGON`.
        vmag_min (int, optional): Minimum Vmag to include, default 4 inclusive.
        vmag_max (int, optional): Maximum Vmag to include, default 17 non-inclusive.
        bq_client (`google.cloud.bigquery.Client`): The BigQuery Client connection.
        **kwargs: Description

    Returns:
        `pandas.DataFrame`: Dataframe containing the results.

    """
    if shape is not None:
        sql_constraint = f"AND ST_CONTAINS(ST_GEOGFROMTEXT('POLYGON(({shape}))'), coords)"

    # Note that for how the BigQuery partition works, we need the parition one step
    # below the requested Vmag_max.
    sql = f"""
    SELECT
        id as picid, twomass, gaia,
        ra as catalog_ra,
        dec as catalog_dec,
        vmag as catalog_vmag,
        vmag_partition as catalog_vmag_bin,
        e_vmag as catalog_vmag_err
    FROM catalog.pic
    WHERE
      vmag_partition BETWEEN {vmag_min} AND {vmag_max - 1}
      {sql_constraint}
    """

    if bq_client is None or bqstorage_client is None:
        bq_client, bqstorage_client = get_bq_clients()

    results = None
    try:
        results = bq_client.query(sql)
        if return_dataframe:
            results = results.result().to_dataframe(bqstorage_client=bqstorage_client)
            logger.debug(f'Found {len(results)} in Vmag=[{vmag_min}, {vmag_max}) and bounds=[{shape}]')
    except Exception as e:
        logger.warning(e)

    return results


def lookup_point_sources(fits_file,
                         catalog_match=False,
                         force_new=False,
                         wcs=None,
                         **kwargs
                         ):
    """Extract point sources from image.

    This function will extract the sources from the image using the given method
    (currently only `source-extractor`). This is returned as a `pandas.DataFrame`. If
    `catalog_match=True` then the resulting sources will be matched against the
    PANOPTES catalog, which is a filtered version of the TESS Input Catalog. See
    `get_catalog_match` for details and column list.

    `source-extractor` will return the following columns:

    * ALPHA_J2000   ->  measured_ra
    * DELTA_J2000   ->  measured_dec
    * XPEAK_IMAGE   ->  measured_x
    * YPEAK_IMAGE   ->  measured_y
    * X_IMAGE       ->  measured_x_image
    * Y_IMAGE       ->  measured_y_image
    * ELLIPTICITY   ->  measured_ellipticity
    * THETA_IMAGE   ->  measured_theta_image
    * FLUX_BEST     ->  measured_flux_best
    * FLUXERR_BEST  ->  measured_fluxerr_best
    * FLUX_MAX      ->  measured_flux_max
    * FLUX_GROWTH   ->  measured_flux_growth
    * MAG_BEST      ->  measured_mag_best
    * MAGERR_BEST   ->  measured_magerr_best
    * FWHM_IMAGE    ->  measured_fwhm_image
    * BACKGROUND    ->  measured_background
    * FLAGS         ->  measured_flags

    .. note::

        Sources within a certain `trim_size` (default 10) of the image edges will be automatically pruned.

    >>> from panoptes.utils.stars import lookup_point_sources
    >>> fits_fn = getfixture('solved_fits_file')

    >>> point_sources = lookup_point_sources(fits_fn)
    >>> point_sources.describe()
             measured_ra    measured_dec  ...    measured_background    measured_flags
    count     473.000000      473.000000  ...             473.000000        473.000000
    mean      303.284052       46.011116  ...            2218.525156          1.143763
    std         0.810261        0.582264  ...               4.545206          3.130030
    min       301.794797       45.038730  ...            2205.807000          0.000000
    25%       302.598079       45.503276  ...            2215.862000          0.000000
    50%       303.243873       46.021710  ...            2218.392000          0.000000
    75%       303.982358       46.497813  ...            2221.577000          0.000000
    max       304.637887       47.015707  ...            2229.050000         27.000000
    ...
    >>> type(point_sources)
    <class 'pandas.core.frame.DataFrame'>

    Args:
        fits_file (str, optional): Path to FITS file to search for stars.
        catalog_match (bool, optional): If `get_catalog_match` should be called after
            looking up sources. Default False. If True, the `args` and `kwargs` will
            be passed to `get_catalog_match`.
        wcs (`astropy.wcs.WCS`|None): A WCS file to use. Default is `None`, in which
            the WCS comes from the `fits_file`.
        force_new (bool, optional): Force a new catalog to be created,
            defaults to False.
        **kwargs: Passed to `get_catalog_match` when `catalog_match=True`.

    Raises:
        Exception: Raised for any exception.

    Returns:
        `pandas.DataFrame`: A dataframe contained the sources.

    """
    if catalog_match:
        if wcs is None:
            wcs = fits_utils.getwcs(fits_file)
        assert wcs is not None and wcs.is_celestial, logger.warning("Need a valid WCS")

    logger.debug(f"Looking up sources for {fits_file}")

    try:
        logger.debug(f"Running source-extractor on {fits_file}")
        point_sources = extract_sources(fits_file, force_new=force_new, **kwargs)
    except Exception as e:
        raise Exception(f"Problem with source-extractor: {e!r} {fits_file}")

    if catalog_match:
        logger.debug(f'Doing catalog match against stars for fits_file={fits_file}')
        try:
            point_sources = get_catalog_match(point_sources, wcs=wcs, **kwargs)
            logger.debug(f'Done with catalog match for {fits_file}')
        except Exception as e:  # pragma: no cover
            logger.error(f'Error in catalog match, returning unmatched results: {e!r} {fits_file}')

    logger.debug(f'Point sources: {len(point_sources)} {fits_file}')
    return point_sources


def get_catalog_match(point_sources,
                      wcs=None,
                      catalog_stars=None,
                      max_separation_arcsec=None,
                      return_unmatched=False,
                      ra_column='measured_ra',
                      dec_column='measured_dec',
                      origin=1,
                      **kwargs):
    """Match the point source positions to the catalog.

    The catalog is matched to the PANOPTES Input Catalog (PIC), which is derived
    from the [TESS Input Catalog](https://tess.mit.edu/science/tess-input-catalogue/)
    [v8](https://heasarc.gsfc.nasa.gov/docs/tess/tess-input-catalog-version-8-tic-8-is-now-available-at-mast.html).

    The catalog is stored in a BigQuery dataset. This function will match the
    `measured_ra` and `measured_dec` columns (as output from `lookup_point_sources`)
    to the `ra` and `dec` columns of the catalog.  The actual lookup is done via
    the `get_stars_from_footprint` function.

    The columns are added to `point_sources`, which is then returned to the user.

    Columns that are added to `point_sources` include:

        * picid
        * unit_id
        * camera_id
        * time
        * gaia
        * twomass
        * catalog_dec
        * catalog_ra
        * catalog_sep_arcsec
        * catalog_measured_diff_arcsec_dec
        * catalog_measured_diff_arcsec_ra
        * catalog_measured_diff_x
        * catalog_measured_diff_y
        * catalog_vmag
        * catalog_vmag_err
        * catalog_x
        * catalog_y
        * catalog_x_int
        * catalog_y_int

    Note:

        Note all fields are expected to have values. In particular, the `gaia`
        and `twomass` fields are often mutually exclusive.  If `return_unmatched=True`
        (see below) then all values related to matching will be `NA` for all `source-extractor`
        related columns.

    By default only the sources that are successfully matched by the catalog are returned.
    This behavior can be changed by setting `return_unmatched=True`. This will append
    *all* catalog entries within the Vmag range [vmag_min, vmag_max).

    Warning:

        Using `return_unmatched=True` can return a very large datafraame depending
        on the chosen Vmag range and galactic coordinates. However, it should be
        noted that limiting the Vmag range makes results less accurate.

        The best policy would be to try to minimize calls to this function. The
        resulting dataframe can be saved locally with `point_sources.to_csv(path_name)`.

    If a `max_separation_arcsec` is given then results will be filtered if their
    match with `source-extractor` was larger than the number given. Typical values would
    be in the range of 20-30 arcsecs, which corresponds to 2-3 pixels.

    Returns:
        `pandas.DataFrame`: A dataframe with the catalog information added to the
        sources.

    Args:
        point_sources (`pandas.DataFrame`): The DataFrame containing point sources
            to be matched. This usually comes from the output of `lookup_point_sources`
            but could be done manually.
        wcs (`astropy.wcs.WCS`, optional): The WCS instance to use for the catalog lookup.
            Either the `wcs` or the `catalog_stars` must be supplied.
        catalog_stars (`pandas.DataFrame`, optional): If provided, the catalog match
            will be performed against this set of stars rather than performing a lookup.
        ra_column (str): The column name to use for the RA coordinates, default `measured_ra`.
        dec_column (str): The column name to use for the Dec coordinates, default `measured_dec`.
        origin (int, optional): The origin for catalog matching, either 0 or 1 (default).
        max_separation_arcsec (float|None, optional): If not None, sources more
            than this many arcsecs from catalog will be filtered.
        return_unmatched (bool, optional): If all results from catalog should be
            returned, not just those with a positive match.
        origin (int): The origin for the WCS. See `all_world2pix`. Default 1.
        **kwargs: Extra options are passed to `get_stars_from_wcs`, which passes them to `get_stars`.

    """
    assert point_sources is not None

    if catalog_stars is None:
        logger.debug(f'Looking up stars for wcs={wcs.wcs.crval}')
        # Lookup stars in catalog
        catalog_stars = get_stars_from_wcs(wcs, **kwargs)

    if catalog_stars is None:
        logger.debug('No catalog matches, returning table without ids')
        return point_sources

    # Get the XY positions
    catalog_xy = wcs.all_world2pix(catalog_stars[['catalog_ra', 'catalog_dec']], origin, ra_dec_order=True)
    catalog_stars['catalog_x'] = catalog_xy.T[0]
    catalog_stars['catalog_y'] = catalog_xy.T[1]
    catalog_stars['catalog_x_int'] = catalog_stars.catalog_x.astype(int)
    catalog_stars['catalog_y_int'] = catalog_stars.catalog_y.astype(int)

    # Get coords for catalog stars
    catalog_coords = SkyCoord(
        ra=catalog_stars['catalog_ra'].values * u.deg,
        dec=catalog_stars['catalog_dec'].values * u.deg
    )

    # Get coords from detected point sources
    stars_coords = SkyCoord(
        ra=point_sources[ra_column].values * u.deg,
        dec=point_sources[dec_column].values * u.deg
    )

    # Do catalog matching
    logger.debug(f'Matching {len(catalog_coords)} catalog stars to {len(stars_coords)} detected stars')
    idx, d2d, d3d = match_coordinates_sky(stars_coords, catalog_coords)
    logger.debug(f'Got {len(idx)} matched sources (includes duplicates) for wcs={wcs.wcs.crval}')

    # Add the matches and their separation.
    point_sources = point_sources.join(catalog_stars.iloc[idx].reset_index(drop=True))
    point_sources['catalog_sep_arcsec'] = d2d.to(u.arcsec).value

    # All point sources so far are matched.
    point_sources['status'] = 'matched'

    # Reorder columns so id cols are first then alpha.
    new_column_order = sorted(list(point_sources.columns))
    id_cols = ['picid', 'unit_id', 'camera_id', 'time', 'gaia', 'twomass', 'status']
    for i, col in enumerate(id_cols):
        new_column_order.remove(col)
        new_column_order.insert(i, col)
    point_sources = point_sources.reindex(columns=new_column_order)

    # Sources that didn't match.
    if return_unmatched:
        logger.debug(f'Adding unmatched sources to table for wcs={wcs.wcs.crval!r}')
        unmatched = catalog_stars.iloc[catalog_stars.index.difference(idx)].copy()
        unmatched['status'] = 'unmatched'
        point_sources = point_sources.append(unmatched)

    # Correct some dtypes.
    point_sources.status = point_sources.status.astype('category')

    logger.debug(f'Point sources: {len(point_sources)} for wcs={wcs.wcs.crval!r}')

    # Remove catalog matches that are too far away.
    if max_separation_arcsec is not None:
        logger.debug(f'Removing matches > {max_separation_arcsec} arcsec from catalog.')
        point_sources = point_sources.query('catalog_sep_arcsec <= @max_separation_arcsec')

    logger.debug(f'Returning point sources: {len(point_sources)} for wcs={wcs.wcs.crval!r}')
    return point_sources


def extract_sources(fits_file,
                    measured_params=None,
                    trim_size=10,
                    force_new=False,
                    img_dimensions=(3476, 5208),
                    extractor_config='resources/source-extractor/panoptes.conf',
                    extractor_params='resources/source-extractor/panoptes.params',
                    *args, **kwargs):
    # Write the source-extractor catalog to a file
    base_dir = os.path.dirname(fits_file)
    source_dir = os.path.join(base_dir, 'source-extractor')
    os.makedirs(source_dir, exist_ok=True)

    image_time = os.path.splitext(os.path.basename(fits_file))[0]

    catalog_filename = os.path.join(source_dir, f'point_sources_{image_time}.cat')
    logger.debug(f"Point source catalog: {catalog_filename}")

    if not extractor_config.startswith('/'):
        extractor_config_path = str(get_project_root() / extractor_config)
    else:
        extractor_config_path = str(extractor_config)
    logger.debug(f"Extractor config: {extractor_config_path}")

    if not extractor_params.startswith('/'):
        extractor_params_path = str(get_project_root() / extractor_params)
    else:
        extractor_params_path = str(extractor_params)
    logger.debug(f"Extractor config: {extractor_params_path}")

    if not os.path.exists(catalog_filename) or force_new:
        logger.debug("No catalog found, building from source-extractor")
        # Build catalog of point sources
        source_extractor = shutil.which('sextractor') or shutil.which('source-extractor')

        assert source_extractor is not None, 'source-extractor not found'

        # source-extractor can't handle compressed data.
        if fits_file.endswith('.fz'):
            fits_file = fits_utils.funpack(fits_file)

        measured_params = measured_params or [
            '-c', extractor_config_path,
            '-PARAMETERS_NAME', extractor_params_path,
            '-CATALOG_NAME', catalog_filename,
        ]

        logger.debug("Running source-extractor...")
        cmd = [source_extractor, *measured_params, fits_file]
        logger.debug(cmd)

        try:
            subprocess.run(cmd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           timeout=60,
                           check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Problem running source-extractor: {e.stderr}\n\n{e.stdout}")

    # Read catalog
    logger.debug(f'Building detected source table with {catalog_filename}')
    point_sources = Table.read(catalog_filename, format='ascii.sextractor')

    # Rename columns
    point_sources.rename_column('XPEAK_IMAGE', 'x')
    point_sources.rename_column('YPEAK_IMAGE', 'y')

    # Filter point sources near edge
    # w, h = data[0].shape
    logger.debug('Trimming sources near edge')
    w, h = img_dimensions
    top = point_sources['y'] > trim_size
    bottom = point_sources['y'] < w - trim_size
    left = point_sources['x'] > trim_size
    right = point_sources['x'] < h - trim_size

    # Do the trim an convert to pandas.
    point_sources = point_sources[top & bottom & right & left].to_pandas()

    # Rename all the columns at once.
    point_sources.columns = [
        'measured_ra',
        'measured_dec',
        'measured_x_int',
        'measured_y_int',
        'measured_x',
        'measured_y',
        'measured_ellipticity',
        'measured_theta_image',
        'measured_flux_best',
        'measured_fluxerr_best',
        'measured_flux_max',
        'measured_flux_growth',
        'measured_mag_best',
        'measured_magerr_best',
        'measured_fwhm_image',
        'measured_background',
        'measured_flags',
    ]

    # Add the image id to the front
    unit_id, camera_id, obstime = fits_utils.getval(fits_file, 'IMAGEID').split('_')
    point_sources['unit_id'] = unit_id
    point_sources['camera_id'] = camera_id
    point_sources['time'] = obstime

    logger.debug(f'Returning {len(point_sources)} sources from source-extractor')
    return point_sources.convert_dtypes()
