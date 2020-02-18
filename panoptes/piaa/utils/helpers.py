import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, sigma_clip

from panoptes.utils.images import fits as fits_utils
from panoptes.utils.bayer import get_rgb_data
from panoptes.utils.google.cloudsql import get_cursor

import logging


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

    return get_stars(ra.min(), ra.max(), dec.min(), dec.max(), **kwargs)


def get_stars(
        ra_min,
        ra_max,
        dec_min,
        dec_max,
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
        if not cursor:
            cursor = get_cursor(port=5433, db_name='v702', db_user='panoptes')

    fetch_sql = f"""
        SELECT
            id,
            ra, dec,
            tmag, e_tmag, vmag, e_vmag,
            lumclass, lum, e_lum,
            contratio, numcont
        FROM {table}
        WHERE
            vmag < 13 AND
            ra >= %s AND ra <= %s AND
            dec >= %s AND dec <= %s;
    """

    cursor.execute(fetch_sql, (ra_min, ra_max, dec_min, dec_max))

    if cursor_only:
        return cursor

    # Get column names
    column_names = [desc.name for desc in cursor.description]

    rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=column_names)
