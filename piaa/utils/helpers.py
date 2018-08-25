import os
from warnings import warn

import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.table import Table
from astropy.wcs import WCS
from astropy.visualization import LogStretch, ImageNormalize, LinearStretch

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils import RectangularAperture

from pong.utils import db

from decimal import Decimal
from copy import copy

palette = copy(plt.cm.inferno)
palette.set_over('w', 1.0)
palette.set_under('k', 1.0)
palette.set_bad('g', 1.0)

rc('animation', html='html5')
plt.style.use('bmh')


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


def show_stamps(pscs,
                frame_idx=None,
                stamp_size=11,
                aperture_position=None,
                aperture_size=None,
                show_normal=False,
                show_residual=False,
                stretch=None,
                save_name=None,
                **kwargs):

    if aperture_position is None:
        midpoint = (stamp_size - 1) / 2
        aperture_position = (midpoint, midpoint)

    if aperture_size:
        aperture = RectangularAperture(
            aperture_position, w=aperture_size, h=aperture_size, theta=0)

    ncols = len(pscs)

    if show_residual:
        ncols += 1

    nrows = 1

    fig = Figure()
    FigureCanvas(fig)
    fig.set_dpi(100)
    fig.set_figheight(4)
    fig.set_figwidth(9)

    if frame_idx is not None:
        s0 = pscs[0][frame_idx]
        s1 = pscs[1][frame_idx]
    else:
        s0 = pscs[0]
        s1 = pscs[1]

    if stretch == 'log':
        stretch = LogStretch()
    else:
        stretch = LinearStretch()

    ax1 = fig.add_subplot(nrows, ncols, 1)

    im = ax1.imshow(s0, origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
    if aperture_size:
        aperture.plot(color='r', lw=4, ax=ax1)
        # annulus.plot(color='c', lw=2, ls='--', ax=ax1)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax1.set_title('Target')

    # Comparison
    ax2 = fig.add_subplot(nrows, ncols, 2)
    im = ax2.imshow(s1, origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
    if aperture_size:
        aperture.plot(color='r', lw=4, ax=ax1)
        # annulus.plot(color='c', lw=2, ls='--', ax=ax1)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax2.set_title('Comparison')

    if show_residual:
        ax3 = fig.add_subplot(nrows, ncols, 3)

        # Residual
        im = ax3.imshow((s0 / s1), origin='lower', cmap=palette,
                        norm=ImageNormalize(stretch=stretch))

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        # ax1.set_title('Residual')
        residual = 1 - (s0.sum() / s1.sum())
        ax3.set_title('Residual {:.01%}'.format(residual))

    # Turn off tick labels
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])

    if save_name:
        try:
            fig.savefig(save_name)
        except Exception as e:
            warn("Can't save figure: {}".format(e))

    return fig


def normalize(cube):
    # Helper function to normalize a stamp
    return (cube.T / cube.sum(1)).T


def spiral_matrix(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[:, 0][::-1])  # take first row and reverse it
        A = A[:, 1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)


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


def animate_stamp(d0):

    fig = Figure()
    FigureCanvas(fig)

    ax = fig.add_subplot(111)

    line = ax.imshow(d0[0])

    def animate(i):
        line.set_data(d0[i])  # update the data
        return line,

    # Init only required for blitting to give a clean slate.
    def init():
        line.set_data(d0[0])
        return line,

    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(d0)), init_func=init,
                                  interval=500, blit=True)

    return ani


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


def plot_pixel_drift(x_pos, y_pos, index=None, out_fn=None, title=None):
    """Plot pixel drift.

    Args:
        x_pos (`numpy.array`): an array of pixel values.
        y_pos (`numpy.array`): an array of pixel values.
        index (`numpy.array`): an array to use as index, can be datetime values.
            If no index is provided a simple range is generated.
        out_fn (str): Filename to save image to, default is None for no save.

    Returns:
        `matplotlib.Figure`: The `Figure` object
    """
    # Plot the pixel drift of target
    if index is None:
        index = np.arange(len(x_pos))

    pos_df = pd.DataFrame({'dx': x_pos, 'dy': y_pos}, index=index)

    fig = Figure()
    FigureCanvas(fig)

    fig.set_figwidth(12)
    fig.set_figheight(9)

    ax = fig.add_subplot(111)
    ax.plot(pos_df.index, pos_df.dx, label='dx')
    ax.plot(pos_df.index, pos_df.dy, label='dy')

    ax.set_ylabel('Δ pixel', fontsize=16)
    ax.set_xlabel('Time [UTC]', fontsize=16)

    if title is None:
        title = 'Pixel drift'

    ax.set_title(title, fontsize=18)
    ax.set_ylim([-5, 5])

    fig.legend(fontsize=16)
    fig.tight_layout()

    if out_fn:
        fig.savefig(out_fn, dpi=100)

    return fig


def get_planet_phase(period, midpoint, t):
    """Get planet phase from period and midpoint. """
    return ((Time(t).mjd - Time(midpoint).mjd) % period) / period
