import os
import sys

import numpy as np
from collections import namedtuple

from decimal import Decimal
from decimal import ROUND_HALF_UP

from astropy.time import Time
from astropy.table import Table
from astropy.wcs import WCS

from piaa.utils import postgres as clouddb
from pocs.utils.images import fits as fits_utils

import logging
logger = logging.getLogger(__name__)


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
        cursor = clouddb.get_cursor(
                instance='tess-catalog', 
                db_name='v6', 
                db_user='postgres',
                port=5433,
                **kwargs)
        
    cursor.execute("""SELECT id, ra, dec, tmag, vmag, e_tmag, twomass
        FROM {}
        WHERE tmag < 13 AND ra >= %s AND ra <= %s AND dec >= %s AND dec <= %s;""".format(table),
                (ra_min, ra_max, dec_min, dec_max)
                )
    if cursor_only:
        return cursor

    d0 = cursor.fetchall()
    if verbose:
        print(d0)
    return Table(
        data=d0,
        names=['id', 'ra', 'dec', 'tmag', 'vmag', 'e_tmag', 'twomass'],
        dtype=['i4', 'f8', 'f8', 'f4', 'f4', 'f4', 'U26'])


def get_star_info(picid=None, twomass_id=None, table='full_catalog', cursor=None, raw=False, verbose=False, **kwargs):
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
        cursor = clouddb.get_cursor(db_name='v6', db_user='postgres', **kwargs)

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


def get_rgb_data(data, **kwargs):
    rgb_masks = get_rgb_masks(data, **kwargs)

    assert rgb_masks is not None

    r_data = np.ma.array(data, mask=~rgb_masks['r'])
    g_data = np.ma.array(data, mask=~rgb_masks['g'])
    b_data = np.ma.array(data, mask=~rgb_masks['b'])

    return np.ma.array([r_data, g_data, b_data])


def get_rgb_masks(data, separate_green=False, mask_path=None, force_new=False, verbose=False):
    """Get the RGGB Bayer pattern for the given data.

    Args:
        data (`numpy.array`): An array of data representing an image.
        separate_green (bool, optional): If the two green channels should be separated,
            default False.
        mask_path (str, optional): Path to file to save/lookup mask.
        force_new (bool, optional): If a new file should be generated, default False.
        verbose (bool, optional): Verbose, default False.

    Returns:
        TYPE: Description
    """
    if mask_path is None:
        mask_path = os.path.join(os.environ['PANDIR'], 'rgb_masks.npz')

    logger.debug('Mask path: {}'.format(mask_path))

    if force_new:
        logger.info("Forcing a new mask file")
        try:
            os.remove(mask_path)
        except FileNotFoundError:
            pass

    # Try to load existing file and if not generate new
    try:
        loaded_masks = np.load(mask_path)
        logger.debug("Loaded masks")
        mask_shape = loaded_masks[loaded_masks.files[0]].shape
        if mask_shape != data.shape:
            logger.debug("Removing mask with wrong size")
            os.remove(mask_path)
            raise FileNotFoundError
        else:
            logger.debug("Using saved masks")
            _rgb_masks = {color: loaded_masks[color] for color in loaded_masks.files}
    except FileNotFoundError:
        logger.info("Making RGB masks")

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
            logger.debug("Making separate green masks")
            green1_mask = np.flipud(np.array(
                [(index[0] % 2 == 0 and index[1] % 2 == 1) for index, i in np.ndenumerate(data)]
            ).reshape(w, h))
            green2_mask = np.flipud(np.array(
                [(index[0] % 2 == 1 and index[1] % 2 == 0) for index, i in np.ndenumerate(data)]
            ).reshape(w, h))

            _rgb_masks = {
                'r': red_mask,
                'g': green1_mask,
                'c': green2_mask,
                'b': blue_mask,
            }
        else:
            green_mask = np.flipud(np.array(
                [((index[0] % 2 == 0 and index[1] % 2 == 1) or
                    (index[0] % 2 == 1 and index[1] % 2 == 0))
                    for index, i in np.ndenumerate(data)
                 ]
            ).reshape(w, h))

            _rgb_masks = {
                'r': red_mask,
                'g': green_mask,
                'b': blue_mask,
            }

        logger.info("Saving masks files")
        np.savez_compressed(mask_path, **_rgb_masks)

    return _rgb_masks


def spiral_matrix(A):
    """Simple function to spiral a matrix.

    Args:
        A (`numpy.array`): Array to spiral.

    Returns:
        `numpy.array`: Spiralled array.
    """
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


def get_stamp_slice(x, y, stamp_size=(14, 14), verbose=False, ignore_superpixel=False):
    """Get the slice around a given position with fixed Bayer pattern.

    Given an x,y pixel position, get the slice object for a stamp of a given size
    but make sure the first position corresponds to a red-pixel. This means that
    x,y will not necessarily be at the center of the resulting stamp.

    Args:
        x (float): X pixel position.
        y (float): Y pixel position.
        stamp_size (tuple, optional): The size of the cutout, default (14, 14).
        verbose (bool, optional): Verbose, default False.

    Returns:
        `slice`: A slice object for the data.
    """
    # Make sure requested size can have superpixels on each side.
    if not ignore_superpixel:
        for side_length in stamp_size:
            side_length -= 2  # Subtract center superpixel
            if int(side_length / 2) % 2 != 0:
                print("Invalid slice size: ", side_length + 2,
                      " Slice must have even number of pixels on each side of",
                      " the center superpixel.",
                      "i.e. 6, 10, 14, 18...")
                return

    # Pixels have nasty 0.5 rounding issues
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

    # Alter the bounds depending on identified center pixel
    if color == 'B':
        x_min -= 1
        x_max -= 1
        y_min -= 1
        y_max -= 1
    elif color == 'G1':
        x_min -= 1
        x_max -= 1
        y_min -= 0
        y_max -= 0
    elif color == 'G2':
        x_min -= 1
        x_max -= 1
        y_min -= 0
        y_max -= 0
    elif color == 'R':
        x_min -= 1
        x_max -= 1
        y_min -= 1
        y_max -= 1

    if verbose:
        print(x_min, x_max, y_min, y_max)

    return (slice(y_min, y_max), slice(x_min, x_max))


def moving_average(data_set, periods=3):
    """Moving average.

    Args:
        data_set (`numpy.array`): An array of values over which to perform the moving average.
        periods (int, optional): Number of periods.

    Returns:
        `numpy.array`: An array of the computed averages.
    """
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='same')


def get_pixel_drift(coords, files):
    """Get the pixel drift for a given set of coordinates.

    Args:
        coords (`astropy.coordinates.SkyCoord`): Coordinates of source.
        files (list): A list of FITS files with valid WCS.

    Returns:
        `numpy.array, numpy.array`: A 2xN array of pixel deltas where
            N=len(files)
    """
    # Get target positions for each frame
    logger.info("Getting pixel drift for {}".format(coords))
    target_pos = list()
    for fn in files:
        h0 = fits_utils.getheader(fn)
        pos = WCS(h0).all_world2pix(coords.ra, coords.dec, 1)
        target_pos.append(pos)

    target_pos = np.array(target_pos)

    # Subtract out the mean to get just the pixel deltas
    x_pos = target_pos[:, 0]
    y_pos = target_pos[:, 1]

    x_pos -= x_pos.mean()
    y_pos -= y_pos.mean()

    return x_pos, y_pos


def get_planet_phase(period, midpoint, obs_time):
    """Get planet phase from period and midpoint.

    Args:
        period (float): The length of the period in days.
        midpoint (`datetime.datetime`): The midpoint of the transit.
        obs_time (`datetime.datetime`): The time at which to compute the phase.

    Returns:
        float: The phase of the planet.
    """
    return ((Time(obs_time).mjd - Time(midpoint).mjd) % period) / period

def scintillation_index(exptime, airmass, elevation, diameter=0.061, scale_height=8000, correction_coeff=1.5):
    """Calculate the scintillation index.
    
    A modification to Young's approximation for estimating the scintillation index, this
    uses a default correction coefficient of 1.5 (see reference).
    
    Note:
        The scintillation index defines the amount of scintillation and is expressed as a variance. 
        Scintillation noise is the square root of the index value.
    
    Empirical Coefficients:
        Observatory Cmedian C Q1  CQ3
        Armazones      1.61 1.30 2.00
        La Palma       1.30 1.02 1.62
        Mauna Kea      1.63 1.34 2.02
        Paranal        1.56 1.27 1.90
        San Pedro      1.67 1.32 2.14
        Tololo         1.42 1.17 1.74    

    For PANOPTES, the default lens is an 85 mm f/1.4 lens. This gives an effective
    diameter of:
        # 85 mm at f/1.4
        diameter = 85 / 1.4 
        diameter = 0.060714286 m
        
    Reference:
        Osborn, J., Föhring, D., Dhillon, V. S., & Wilson, R. W. (2015). 
        Atmospheric scintillation in astronomical photometry. 
        Monthly Notices of the Royal Astronomical Society, 452(2), 1707–1716. 
        https://doi.org/10.1093/mnras/stv1400        
        
    """
    zenith_distance = (np.arccos(1 / airmass))
    
    return 10e-6 * (correction_coeff**2) * \
            (diameter**(-4/3)) * \
            (1/exptime) * \
            (np.cos(zenith_distance)**-3) * \
            np.exp(-2*elevation / scale_height)

