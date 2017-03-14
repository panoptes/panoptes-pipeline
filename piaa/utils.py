import numpy as np

from decimal import Decimal
from decimal import ROUND_HALF_UP

from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.wcs import WCS

from matplotlib import pyplot as plt
from photutils import RectangularAperture


class StampSizeException(Exception):
    pass


def show_aperture_stamps(seq_files, point_sources):
    fig, ax = plt.subplots(6, 5)
    fig.set_size_inches(15, 22)

    # sns.set_style('white')

    for f in range(6):
        for i in range(5):

            d0 = fits.getdata(seq_files[f])
            wcs = WCS(seq_files[f])

            loc = point_sources.iloc[i]

            coords = wcs.all_world2pix(loc['ALPHA_J2000'], loc['DELTA_J2000'], 1)
    #         coords = wcs.all_world2pix(target.ra.value, target.dec.value, 1)

            color = pixel_color(coords[0], coords[1])
            # Get large stamp
            c0 = make_postage_stamp(d0, coords[0], coords[1], radius=5)

            # Find centroid
            x1 = 4.5
            y1 = 4.5

            ax[f][i].imshow(c0)
            #ax[f][i].plot(coords[0] % 10, coords[1] % 10, color='red', marker='+', ms=3, mew=30)
            ax[f][i].set_title("Image: {} Ref: {} - {}".format(f, loc.name, color))
            apertures = RectangularAperture((x1, y1), w=6, h=6, theta=0)
            # annulus = RectangularAnnulus((x1, y1), w_in=4, w_out=10, h_out=10, theta=0)
            apertures.plot(color='b', lw=1.5, ax=ax[f][i])
    #         annulus.plot(color='b', lw=2, ax=ax[f][i])
            # ax[f][i].set_title(color)


def get_index(x):
    """ Find corresponding index position of `x` pixel position

    Note:
        Due to the standard rounding policy of python that will round half integers
        to their nearest even whole integer, we instead use a `Decimal` with correct
        round up policy.

    Args:
        x (float): x coordinate position

    Returns:
        int: Index position for zero-based index
    """
    return int(Decimal(x - 1).to_integral_value(ROUND_HALF_UP))


def pixel_color(col, row, zero_based=False):
    if zero_based:
        assert isinstance(col, int), "Index mode only accepts integers"
        assert isinstance(row, int), "Index mode only accepts integers"
        assert row >= 0 and row < 3476, 'Row value outside dimensions of image'
        assert col >= 0 and col < 5208, 'Column value outside dimensions of image'
    else:
        assert row >= 0.5 and row < 3476.5, 'Row value outside dimensions of image'
        assert col >= 0.5 and col < 5208.5, 'Column value outside dimensions of image'

    row = get_index(row)
    col = get_index(col)

    if row < 0:
        row = 0
    if col < 0:
        col = 0

    color = None

    if (row % 2 == 1) and (col % 2 == 0):
        color = 'R'

    if (row % 2 == 1) and (col % 2 == 1):
        color = 'G1'

    if (row % 2 == 0) and (col % 2 == 0):
        color = 'G2'

    if (row % 2 == 0) and (col % 2 == 1):
        color = 'B'

    return color


def make_slice(x, y, radius=3, padding=None):
    """

    Center of stamp:
        G B
        R G


    """
    if radius > 2:
        assert radius % 2 == 1, "Radius must be odd number so full length is pixel multiple"

    if padding is not None:
        assert isinstance(padding, tuple)
        assert len(padding) == 4
    else:
        padding = (0, 0, 0, 0)

    color = pixel_color(x, y, zero_based=False)

    x_idx = get_index(x)
    y_idx = get_index(y)

    # Correct so Red pixel is always in lower-left
    if color == 'R':
        center_x = x_idx
        center_y = y_idx
    elif color == 'B':
        center_x = x_idx
        center_y = y_idx
    elif color == 'G1':
        center_x = x_idx
        center_y = y_idx
    elif color == 'G2':
        center_x = x_idx
        center_y = y_idx

    center_x = int(center_x)
    center_y = int(center_y)

    top = center_y + radius
    bottom = center_y - radius
    left = center_x - radius + 1
    right = center_x + radius + 1

    top -= padding[0]
    right += padding[1]
    bottom += padding[2]
    left -= padding[3]

    return (np.s_[bottom:top], np.s_[left:right])


def make_postage_stamp(data, *args, **kwargs):
    row_slice, column_slice = make_slice(*args, **kwargs)

    if data.ndim == 3:
        return data[:, row_slice, column_slice]
    else:
        return data[row_slice, column_slice]


def make_masks(data):
    if data.ndim > 2:
        data = data[0]

    w, h = data.shape

    red_mask = np.flipud(np.array(
        [index[0] % 2 == 0 and index[1] % 2 == 0 for index, i in np.ndenumerate(data)]
    ).reshape(w, h))

    blue_mask = np.flipud(np.array(
        [index[0] % 2 == 1 and index[1] % 2 == 1 for index, i in np.ndenumerate(data)]
    ).reshape(w, h))

    green_mask = np.flipud(np.array(
        [(index[0] % 2 == 0 and index[1] % 2 == 1) or (index[0] % 2 == 1 and index[1] % 2 == 0)
         for index, i in np.ndenumerate(data)]
    ).reshape(w, h))

    return red_mask, green_mask, blue_mask


def get_fov_plot(ra, dec, width=15, height=10, org=0):
    """ Get points for rectangle corresponding to FOV centered around ra, dec """
    x = np.remainder(ra + (360 * u.degree) - org, (360 * u.degree))  # shift RA values
    ind = x > (180 * u.degree)
    x[ind] -= (360 * u.degree)    # scale conversion to [-180, 180]
    x = -x    # reverse the scale: East to the left

    ra = Angle(x)
#     ra = ra.wrap_at(180 * u.degree)
    dec = Angle(dec)

    width = width * u.degree
    height = height * u.degree

    ra_bl = ra - (width / 2)
    ra_br = ra + (width / 2)
    ra_tl = ra - (width / 2)
    ra_tr = ra + (width / 2)
    x = np.array([ra_bl.radian, ra_br.radian, ra_tr.radian, ra_tl.radian, ra_bl.radian])

    dec_bl = dec - (height / 2)
    dec_br = dec - (height / 2)
    dec_tl = dec + (height / 2)
    dec_tr = dec + (height / 2)
    y = np.array([dec_bl.radian, dec_br.radian, dec_tr.radian, dec_tl.radian, dec_bl.radian])

    return x, y
