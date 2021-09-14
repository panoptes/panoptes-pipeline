import numpy as np
import pandas
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from panoptes.pipeline.utils.gcp.bigquery import get_bq_clients


def get_stars_from_coords(ra: float, dec: float, radius: float = 8.0, **kwargs) -> pandas.DataFrame:
    limits = dict(
        ra_max=ra + radius,
        ra_min=ra - radius,
        dec_max=dec + radius,
        dec_min=dec - radius,
    )

    print(f'Using {limits=} for get_stars')
    catalog_stars = get_stars(shape=limits, **kwargs)

    return catalog_stars


def get_stars_from_wcs(wcs0: WCS, round_to: int = 0, pad: float = 1.0, pad_size=(10, 10),
                       **kwargs) -> pandas.DataFrame:
    """Lookup star information from WCS footprint.

    Generates the correct layout for an SQL `POLYGON` that can be passed to
    :py:func:`get_stars`.

    Args:
        wcs0 (astropy.wcs.WCS): A valid (i.e. `wcs.is_celestial`) World Coordinate System object.
        round_to (int): Round the limits to this decimal place, default 0. Helps with automatic
            bigquery caching by making the query the same each time.
        pad (float): The amount of padding in degrees to add to each of the RA and Dec
            limits, default 0.5 [degrees].
        **kwargs: Optional keywords to pass to :py:func:`get_stars`.

    """
    wcs_footprint = wcs0.calc_footprint()
    print(f'Looking up catalog stars for WCS: {wcs_footprint}')

    ra_max, dec_max = (wcs0.wcs.crval + np.array(pad_size)).round(round_to)
    ra_min, dec_min = (wcs0.wcs.crval - np.array(pad_size)).round(round_to)

    limits = dict(
        ra_max=ra_max % 360,
        ra_min=ra_min % 360,
        dec_max=dec_max,
        dec_min=dec_min
    )

    print(f'Searching square shape with {round_to=} and {pad=}: {limits!r}')
    catalog_stars = get_stars(shape=limits, **kwargs)

    return catalog_stars


def get_stars(
        shape=None,
        vmag_min=7,
        vmag_max=14,
        bq_client=None,
        bqstorage_client=None,
        column_mapping=None,
        return_dataframe=True,
        **kwargs):
    """Look star information from the TESS catalog.

    https://outerspace.stsci.edu/display/TESS/TIC+v8+and+CTL+v8.xx+Data+Release+Notes

    Args:
        shape (dict): A dictionary containing the keys `ra_min`, `ra_max`, `dec_min`, `dec_max`.
        vmag_min (int, optional): Minimum Vmag to include, default 4 inclusive.
        vmag_max (int, optional): Maximum Vmag to include, default 17 non-inclusive.
        bq_client (`google.cloud.bigquery.Client`): The BigQuery Client connection.
        **kwargs: Description

    Returns:
        `pandas.DataFrame`: Dataframe containing the results.

    """
    column_mapping = column_mapping or {
        "id": "picid",
        "gaia": "gaia",
        "ra": "catalog_ra",
        "dec": "catalog_dec",
        "vmag": "catalog_vmag",
        "vmag_partition": "catalog_vmag_bin",
        "e_vmag": "catalog_vmag_err",
        "tmag": "catalog_tmag",
        "e_tmag": "catalog_tmag_err",
        "gaiamag": "catalog_gaiamag",
        "e_gaiamag": "catalog_gaiamag_err",
        # NOTE: The columns in BQ are currently mis-named for the GAIA b and r.
        # The magnitude and error columns are switched, so we trick it here.
        # TODO: Fix the BQ mapping.
        "gaiabp": "catalog_gaiabp_err",
        "e_gaiabp": "catalog_gaiabp",
        "gaiarp": "catalog_gaiarp_err",
        "e_gaiarp": "catalog_gaiarp",
        "numcont": "catalog_numcont",
        "contratio": "catalog_contratio"
    }

    column_mapping_str = ', '.join([f'{k} as {v}' for k, v in column_mapping.items()])

    # The Right Ascension can wrap around from 360° to 0°, so we have to specifically check.
    if shape['ra_max'] < shape['ra_min']:
        ra_constraint = 'OR'
    else:
        ra_constraint = 'AND'

    # Note that for how the BigQuery partition works, we need the partition one step
    # below the requested Vmag_max.
    sql = f"""
    SELECT {column_mapping_str} 
    FROM catalog.pic
    WHERE
        (dec >= {shape['dec_min']} AND dec <= {shape['dec_max']}) AND
        (ra >= {shape['ra_min']} {ra_constraint} ra <= {shape['ra_max']}) AND
        (vmag_partition BETWEEN {vmag_min} AND {vmag_max - 1})
    """

    print(f'{sql=}')

    if bq_client is None or bqstorage_client is None:
        bq_client, bqstorage_client = get_bq_clients()

    results = None
    try:
        results = bq_client.query(sql)
        if return_dataframe:
            results = results.result().to_dataframe()
            print(
                f'Found {len(results)} in Vmag=[{vmag_min}, {vmag_max}) and bounds=[{shape}]')
    except Exception as e:
        print(e)

    return results


def get_catalog_match(point_sources,
                      wcs=None,
                      catalog_stars=None,
                      max_separation_arcsec=None,
                      ra_column='measured_ra',
                      dec_column='measured_dec',
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
        print(f'Looking up stars for wcs={wcs.wcs.crval}')
        # Lookup stars in catalog
        catalog_stars = get_stars_from_wcs(wcs, **kwargs)

    if catalog_stars is None:
        print('No catalog matches, returning table without ids')
        return point_sources

    # Get coords for catalog stars
    catalog_coords = SkyCoord(
        ra=catalog_stars['catalog_ra'].values * u.deg,
        dec=catalog_stars['catalog_dec'].values * u.deg,
        frame='icrs'
    )

    # Get coords from detected point sources
    stars_coords = SkyCoord(
        ra=point_sources[ra_column].values * u.deg,
        dec=point_sources[dec_column].values * u.deg,
        frame='icrs'
    )

    # Do catalog matching
    print(
        f'Matching {len(catalog_coords)} catalog stars to {len(stars_coords)} detected stars')
    idx, d2d, d3d = stars_coords.match_to_catalog_sky(catalog_coords)
    print(f'Got {len(idx)} matched sources (includes duplicates) for wcs={wcs.wcs.crval}')

    catalog_matches = catalog_stars.iloc[idx].copy()
    catalog_matches['catalog_sep'] = d2d.to_value(u.arcsec)

    # Get the XY positions
    catalog_matches = get_xy_positions(wcs, catalog_matches)

    # Add the matches and their separation.
    matched_sources = point_sources.reset_index(drop=True).join(
        catalog_matches.reset_index(drop=True))

    # All point sources so far are matched.
    # matched_sources['status'] = 'matched'
    # matched_sources.status = matched_sources.status.astype('category')

    # Sources that didn't match.
    #     if return_unmatched:
    #         print(f'Adding unmatched sources to table for wcs={wcs.wcs.crval!r}')
    #         unmatched = catalog_stars.iloc[catalog_stars.index.difference(idx)].copy()

    #         unmatched['status'] = 'unmatched'
    #         point_sources = point_sources.append(unmatched)

    # Reorder columns so id cols are first then alpha.
    # new_column_order = sorted(list(matched_sources.columns))
    # id_cols = ['picid', 'unit_id', 'camera_id', 'time', 'gaia', 'twomass', 'status']
    # for i, col in enumerate(id_cols):
    #     new_column_order.remove(col)
    #     new_column_order.insert(i, col)
    # matched_sources = matched_sources.reindex(columns=new_column_order)

    print(f'Point sources: {len(matched_sources)} for wcs={wcs.wcs.crval!r}')

    # Remove catalog matches that are too far away.
    if max_separation_arcsec is not None:
        print(f'Removing matches > {max_separation_arcsec} arcsec from catalog.')
        matched_sources = matched_sources.query('catalog_sep <= @max_separation_arcsec')

    print(f'Returning matched sources: {len(matched_sources)} for wcs={wcs.wcs.crval!r}')
    return matched_sources


def get_xy_positions(wcs_input, catalog_df, ra_column='catalog_ra', dec_column='catalog_dec',
                     origin=1, copy_catalog=True):
    if copy_catalog:
        catalog_df = catalog_df.copy()

    coords = catalog_df[[ra_column, dec_column]]

    # Get the XY positions
    catalog_xy = wcs_input.all_world2pix(coords, origin, ra_dec_order=True)
    catalog_df['catalog_wcs_x'] = catalog_xy.T[0]
    catalog_df['catalog_wcs_y'] = catalog_xy.T[1]
    catalog_df['catalog_wcs_x_int'] = catalog_df.catalog_wcs_x.astype(int)
    catalog_df['catalog_wcs_y_int'] = catalog_df.catalog_wcs_y.astype(int)

    return catalog_df
