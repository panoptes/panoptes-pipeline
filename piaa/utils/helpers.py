import os

import numpy as np

from decimal import Decimal
from decimal import ROUND_HALF_UP

from astropy.time import Time
from astropy.table import Table
from astropy.wcs import WCS

from pong.utils import db


def get_stars_from_footprint(wcs_footprint, **kwargs):
    ra = wcs_footprint[:, 0]
    dec = wcs_footprint[:, 1]

    return get_stars(ra.min(), ra.max(), dec.min(), dec.max(), **kwargs)


def get_stars(
        ra_min,
        ra_max,
        dec_min,
        dec_max,
        table='full_catalog',
        cursor_only=True,
        verbose=False,
        *args,
        **kwargs):
    cur = db.get_cursor(instance='tess-catalog', db='v6', **kwargs)
    cur.execute('SELECT id, ra, dec, tmag, vmag, e_tmag, twomass FROM {} WHERE tmag < 13 AND ra >= %s AND ra <= %s AND dec >= %s AND dec <= %s;'.format(
        table), (ra_min, ra_max, dec_min, dec_max))

    if cursor_only:
        return cur

    d0 = np.array(cur.fetchall())
    if verbose:
        print(d0)
    return Table(
        data=d0,
        names=['id', 'ra', 'dec', 'tmag', 'vmag', 'e_tmag', 'twomass'],
        dtype=['i4', 'f8', 'f8', 'f4', 'f4', 'f4', 'U26'])


def get_star_info(picid=None, twomass_id=None, table='full_catalog', verbose=False, **kwargs):
    cur = db.get_cursor(instance='tess-catalog', db='v6', **kwargs)

    if picid:
        val = picid
        col = 'id'
    elif twomass_id:
        val = twomass_id
        col = 'twomass'

    cur.execute('SELECT * FROM {} WHERE {}=%s'.format(table, col), (val,))
    return cur.fetchone()


def get_rgb_masks(data, separate_green=False, force_new=False, verbose=False):

    rgb_mask_file = 'rgb_masks.npz'

    if force_new:
        try:
            os.remove(rgb_mask_file)
        except FileNotFoundError:
            pass

    try:
        return np.load(rgb_mask_file)
    except FileNotFoundError:
        if verbose:
            print("Making RGB masks")

        if data.ndim > 2:
            data = data[0]

        w, h = data.shape

        red_mask = np.flipud(np.array(
            [index[0] % 2 == 0 and index[1] % 2 == 0 for index, i in np.ndenumerate(data)]
        ).reshape(w, h))

        blue_mask = np.flipud(np.array(
            [index[0] % 2 == 1 and index[1] % 2 == 1 for index, i in np.ndenumerate(data)]
        ).reshape(w, h))

        if separate_green:
            green1_mask = np.flipud(np.array(
                [(index[0] % 2 == 0 and index[1] % 2 == 1) for index, i in np.ndenumerate(data)]
            ).reshape(w, h))
            green2_mask = np.flipud(np.array(
                [(index[0] % 2 == 1 and index[1] % 2 == 0) for index, i in np.ndenumerate(data)]
            ).reshape(w, h))

            _rgb_masks = np.array([red_mask, green1_mask, green2_mask, blue_mask])
        else:
            green_mask = np.flipud(np.array(
                [(index[0] % 2 == 0 and index[1] % 2 == 1) or (index[0] % 2 == 1 and index[1] % 2 == 0)
                 for index, i in np.ndenumerate(data)]
            ).reshape(w, h))

            _rgb_masks = np.array([red_mask, green_mask, blue_mask])

        _rgb_masks.dump(rgb_mask_file)

        return _rgb_masks


def spiral_matrix(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[:, 0][::-1])  # take first row and reverse it
        A = A[:, 1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)


def get_pixel_index(x):
    """Find corresponding index position of `x` pixel position.

    Note:
        Due to the standard rounding policy of python that will round half integers
        to their nearest even whole integer, we instead use a `Decimal` with correct
        round up policy.

    Args:
        x (float): x coordinate position.

    Returns:
        int: Index position for zero-based index
    """
    return int(Decimal(x - 1).to_integral_value(ROUND_HALF_UP))


def pixel_color(x, y):
    """ Given an x,y position, return the corresponding color

    This is a Bayer array with a RGGB pattern in the lower left corner
    as it is loaded into numpy.

    Note:
              0  1  2  3
             ------------
          0 | G2 B  G2 B
          1 | R  G1 R  G1
          2 | G2 B  G2 B
          3 | R  G1 R  G1
          4 | G2 B  G2 B
          5 | R  G1 R  G1

          R : even x, odd y
          G1: odd x, odd y
          G2: even x, even y
          B : odd x, even y

    Returns:
        str: one of 'R', 'G1', 'G2', 'B'
    """
    x = int(x)
    y = int(y)
    if x % 2 == 0:
        if y % 2 == 0:
            return 'G2'
        else:
            return 'R'
    else:
        if y % 2 == 0:
            return 'B'
        else:
            return 'G1'


def get_stamp_slice(x, y, stamp_size=(14, 14), verbose=False):

    for m in stamp_size:
        m -= 2  # Subtract center superpixel
        if int(m / 2) % 2 != 0:
            print("Invalid size: ", m + 2)
            return

    x = Decimal(float(x)).to_integral()
    y = Decimal(float(y)).to_integral()
    color = pixel_color(x, y)
    if verbose:
        print(x, y, color)

    x_half = int(stamp_size[0] / 2)
    y_half = int(stamp_size[1] / 2)

    x_min = int(x - x_half)
    x_max = int(x + x_half)

    y_min = int(y - y_half)
    y_max = int(y + y_half)

    if color == 'B':
        y_min -= 1
        y_max -= 1
    elif color == 'G2':
        x_min -= 1
        x_max -= 1
        y_min -= 1
        y_max -= 1
    elif color == 'R':
        x_min -= 1
        x_max -= 1

    if verbose:
        print(x_min, x_max, y_min, y_max)

    return (slice(y_min, y_max), slice(x_min, x_max))


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='same')


def get_pixel_drift(coords, files, ext=0):
    """Get the pixel drift for a given set of coordinates.

    Args:
        coords (`astropy.coordinates.SkyCoord`): Coordinates of source.
        files (list): A list of FITS files with valid WCS.

    Returns:
        `numpy.array, numpy.array`: A 2xN array of pixel deltas where
            N=len(files)
    """
    # Get target positions for each frame
    if files[0].endswith('fz'):
        ext = 1

    target_pos = np.array([
        WCS(fn, naxis=ext).all_world2pix(coords.ra, coords.dec, 0)
        for fn in files
    ])

    # Subtract out the mean to get just the pixel deltas
    x_pos = target_pos[:, 0]
    y_pos = target_pos[:, 1]

    x_pos -= x_pos.mean()
    y_pos -= y_pos.mean()

    return x_pos, y_pos


def get_planet_phase(period, midpoint, t):
    """Get planet phase from period and midpoint. """
    return ((Time(t).mjd - Time(midpoint).mjd) % period) / period
