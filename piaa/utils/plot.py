import os
import numpy as np
import pandas as pd
from copy import copy
from warnings import warn
from collections import defaultdict

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib import gridspec
from cycler import cycler as cy

from astropy.coordinates import Angle
from astropy.modeling import models, fitting
from astropy import units as u
from astropy.visualization import LogStretch, ImageNormalize, LinearStretch, PercentileInterval, MinMaxInterval
from astropy.stats import sigma_clip
from photutils import RectangularAperture

from piaa.utils import helpers

rc('animation', html='html5')


def get_palette(cmap='inferno'):
    """Get a palette for drawing.

    Returns a copy of the colormap palette with bad pixels marked.

    Args:
        cmap (str, optional): Colormap to use, default 'inferno'.

    Returns:
        `matplotlib.cm`: The colormap.
    """
    palette = copy(getattr(cm, cmap))
    palette.set_over('w', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('g', 1.0)
    return palette

def get_labelled_style_cycler(cmap='viridis'):

    try:
        cmap_colors = cm.get_cmap(cmap).colors
    except ValueError:
        raise Exception(f'Invalid colormap {cmap}')

    cyl = cy('c', cmap_colors)

    finite_cy_iter = iter(cyl)
    styles = defaultdict(lambda : next(finite_cy_iter))

    return styles


def show_stamps(pscs,
                frame_idx=None,
                stamp_size=11,
                aperture_position=None,
                aperture_size=None,
                show_residual=False,
                stretch=None,
                save_name=None,
                show_max=False,
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
    #fig.set_dpi(100)
    fig.set_figheight(4)
    fig.set_figwidth(8)

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

    norm = ImageNormalize(s0, interval=MinMaxInterval(), stretch=stretch)

    ax1 = fig.add_subplot(nrows, ncols, 1)

    im = ax1.imshow(s0, origin='lower', cmap=get_palette(), norm=norm)
    #add_pixel_grid(ax1, stamp_size, stamp_size, show_superpixel=False)
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
    im = ax2.imshow(s1, origin='lower', cmap=get_palette(), norm=norm)
    #add_pixel_grid(ax2, stamp_size, stamp_size, show_superpixel=False)
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
        residual = s0 / s1
        im = ax3.imshow(residual, origin='lower', cmap=get_palette(), norm=ImageNormalize(residual, interval=MinMaxInterval(), stretch=LinearStretch()))
        #add_pixel_grid(ax3, stamp_size, stamp_size, show_superpixel=False)

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax3.set_title('Residual')
        #ax3.set_title('Residual RMS: {:.01%}'.format(residual))
        ax3.set_yticklabels([])
        ax3.set_xticklabels([])

    # Turn off tick labels
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])

    if save_name:
        try:
            fig.savefig(save_name)
        except Exception as e:
            warn("Can't save figure: {}".format(e))

    return fig


def shift_coord(x, origin=0):
    """Shift galactic coordinates for plotting.

    Args:
        x (int): The array of values to be shifted.
        org (int, optional): The origin around which to shift, default 0°.

    Returns:
        TYPE: Description
    """
    x = np.remainder(x + (360 * u.degree) - origin, (360 * u.degree))  # shift RA values
    ind = x > (180 * u.degree)
    x[ind] -= (360 * u.degree)    # scale conversion to [-180, 180]
    x = -x    # reverse the scale: East to the left
    return x


def get_fov_plot(observation=None, coords=None, width=15, height=10, org=0, return_polygon=False):
    """ Get points for rectangle corresponding to FOV centered around ra, dec """

    from shapely.geometry import Polygon

    if coords is not None:
        ra = shift_coord(coords[0], org)
        dec = coords[1]

        ra = Angle(ra)
        dec = Angle(dec)

        width = width * u.degree
        height = height * u.degree

        ra_bl = ra - (width / 2)
        ra_br = ra + (width / 2)
        ra_tl = ra - (width / 2)
        ra_tr = ra + (width / 2)

        dec_bl = dec - (height / 2)
        dec_br = dec - (height / 2)
        dec_tl = dec + (height / 2)
        dec_tr = dec + (height / 2)

    if observation is not None:
        ra, dec = observation.wcs.all_pix2world(
            [0, 0, observation._img_w, observation._img_w],
            [0, observation._img_h, 0, observation._img_h], 0
        )

        ra = [Angle(r * u.degree) for r in ra]
        dec = [Angle(d * u.degree) for d in dec]

        ra_bl = ra[0]
        ra_br = ra[1]
        ra_tl = ra[2]
        ra_tr = ra[3]

        dec_bl = dec[0]
        dec_br = dec[1]
        dec_tl = dec[2]
        dec_tr = dec[3]

    x = np.array([ra_bl.radian, ra_br.radian, ra_tr.radian, ra_tl.radian, ra_bl.radian])
    y = np.array([dec_bl.radian, dec_br.radian, dec_tr.radian, dec_tl.radian, dec_bl.radian])

    if return_polygon:
        polygon = Polygon([
            (ra_bl.value, dec_bl.value),
            (ra_br.value, dec_br.value),
            (ra_tr.value, dec_tr.value),
            (ra_tl.value, dec_tl.value),
        ])
        return x, y, polygon
    else:
        return x, y

    return x, y


def add_pixel_grid(ax1, grid_height, grid_width, show_axis_labels=True, show_superpixel=False,
                   major_alpha=0.5, minor_alpha=0.25):

    # major ticks every 2, minor ticks every 1
    if show_superpixel:
        x_major_ticks = np.arange(-0.5, grid_width, 2)
        y_major_ticks = np.arange(-0.5, grid_height, 2)

        ax1.set_xticks(x_major_ticks)
        ax1.set_yticks(y_major_ticks)

        ax1.grid(which='major', color='r', linestyle='--', lw=3, alpha=major_alpha)
    else:
        ax1.set_xticks([])
        ax1.set_yticks([])

    x_minor_ticks = np.arange(-0.5, grid_width, 1)
    y_minor_ticks = np.arange(-0.5, grid_height, 1)

    ax1.set_xticks(x_minor_ticks, minor=True)
    ax1.set_yticks(y_minor_ticks, minor=True)

    ax1.grid(which='minor', color='r', lw='2', linestyle='--', alpha=minor_alpha)

    if show_axis_labels is False:
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])


def pixel_hist(hdu, save_plot=True, out_fn=None):
    data = hdu.data
    exptime = hdu.header['EXPTIME']

    # Make the rgb masks if needed
    rgb_masks = helpers.get_rgb_masks(data)


    fig = Figure()
    FigureCanvas(fig)

    fig.set_size_inches(15, 7)

    model_fits = list()
    for i, color in enumerate(rgb_masks.keys()):
        ax = fig.add_subplot(1, 3, i + 1)

        mask = rgb_masks[color]
        d0 = np.ma.array(data, mask=~mask)

        d1 = sigma_clip(d0.compressed(), iters=2)
        
        bins = np.arange(d1.min(), d1.max(), 25)
        x = bins[:-1]

        d2, _ = np.histogram(d1, bins=bins)

        y = d2

        # Fit the data using a Gaussian
        g_init = models.Gaussian1D(amplitude=y.max(), mean=d1.mean(), stddev=d1.std())
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, x, y)

        ax.hist(d1, bins=bins, histtype='bar', alpha=0.6)

        # Plot the models
        ax.plot(x, y, 'ko', label='Pixel values', alpha=0.6)
        ax.plot(x, g(x), label='Gaussian', c='k')

        # Plot the mean
        ax.axvline(g.mean.value, ls='--', alpha=0.75, c='k')

        ax.set_title('{} μ={:.02f} σ={:.02f}'.format(
            color.capitalize(), g.mean.value, g.stddev.value))
        ax.set_xlabel('{} Counts [adu]'.format(color.capitalize()))

        if i == 0:
            ax.set_ylabel('Number of pixels')

        model_fits.append(g)

    fig.suptitle('Average pixel counts - {} s exposure'.format(exptime), fontsize=20)

    if out_fn is None and save_plot:
        out_fn = os.path.splitext(hdu.header['FILENAME'])[0] + '_pixel_counts.png'
        fig.savefig(out_fn, dpi=100)

    return model_fits


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
    ax.plot(pos_df.index, pos_df.dx, lw=4, label='dx [Dec axis]')
    ax.plot(pos_df.index, pos_df.dy, lw=4, alpha=0.5, label='dy [RA axis]')

    ax.set_ylabel('Δ pixel', fontsize=18)
    ax.set_xlabel('Time [UTC]', fontsize=16)

    if title is None:
        title = 'Pixel drift'

    ax.set_title(title, fontsize=16)
    ax.set_ylim([-5, 5])

    fig.legend(fontsize=16)
    fig.tight_layout()

    if out_fn:
        fig.savefig(out_fn, dpi=100)

    return fig


def make_apertures_plot(apertures, title=None, num_frames=None, output_dir=None):
    """Make a plot of the final stamp aperture.

    Args:
        apertures (list): List of data to plot.
        num_frames (int, optional): Number of frames to plot, default len(apertures).
    """
    num_cols = 3

    if num_frames is None:
        num_frames = (len(apertures) // num_cols)

    c_lookup = {
        0: 'red',
        1: 'green',
        2: 'blue'
    }

    for row_num in range(num_frames):
        fig = Figure()
        FigureCanvas(fig)
        fig.set_size_inches(9, 3)
        #fig.set_size_inches(num_cols + 1, num_frames)

        axes = fig.subplots(1, num_cols + 1, sharex=True, sharey=True)

        all_channels = None
        
        # One column for each color channel plus the sum
        for col_num in range(num_cols):
            idx = (row_num * (num_cols)) + col_num
            target = apertures[idx][0]
            if all_channels is None:
                all_channels = np.zeros_like(target.filled(0))

            all_channels = all_channels + target.filled(0)

        for col_num in range(num_cols):

            ax = axes[col_num]

            idx = (row_num * (num_cols)) + col_num

            target = apertures[idx][0]
            
            try:
                title = apertures[idx][1]
            except IndexError:
                pass

            im = ax.imshow(target, vmin=target.min(), vmax=target.max(), origin='lower')
                

            # Show the sum
            if col_num == 2:
                ax2 = axes[col_num + 1]
                im = ax2.imshow(all_channels, vmin=all_channels.min(), vmax=all_channels.max(), origin='lower')

                divider = make_axes_locatable(ax2)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)
                
                plt.setp(ax2.get_xticklabels(), visible=False)
                plt.setp(ax2.get_yticklabels(), visible=False)

            # If first row, set column title
            #if row_num == 0:
            ax.set_title(c_lookup[col_num])
            if col_num == 2:
                ax2.set_title('All')
                    
            # If first column, show frame index to left
            #if col_num == 0:
#                 y_lab = ax.set_ylabel()
#                 y_lab.set_rotation(0)

            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

            if title:
                frame_num = "{:03d}".format(int(idx // 3))
                title = f'{frame_num} - {title}'
                try:
                    center_color = apertures[idx][2]
                    title = f'{title} - Center: {center_color}'
                except IndexError:
                    pass
                fig.suptitle(title)
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.1)
            aperture_fn = os.path.join(output_dir, f'{row_num:03d}.png')
            fig.savefig(aperture_fn)


def plot_lightcurve(x, y, model_flux=None, use_imag=False, transit_info=None, color='k', **kwargs):
    """Plot the lightcurve

    Args:
        x (`numpy.array`): X values, usually the index of the lightcurve DataFrame.
        y (`numpy.array`): Y values, flux or magnitude.
        model_flux (`numpy.array`): An array of flux values to act as a model.
            This could also be used to plot a fit line. Default None.
        use_imag (bool): If instrumental magnitudes should be used instead of flux,
            default False.
        transit_info (tuple): A tuple with midpoint, ingress, and egress values.
            Should be in the same formac as the `lc0.index`.
        color (str, optional): Color to be used for main data points, default black.
        **kwargs: Can include the `title` and `ylim`.

    Returns:
        TYPE: Description
    """
    fig = Figure()
    FigureCanvas(fig)

    fig.set_size_inches(12, 9)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    # Lightcurve Plot #

    ax1 = fig.add_subplot(gs[0])

    # Raw data values
    ax1.plot(x, y, marker='o', ls='', color=color, label='images')

    # Transit model
    if model_flux is not None:
        ax1.plot(x, model_flux, c='r', lw=3, ls='--', label='Model transit')

    # Transit lines
    if transit_info is not None:
        midpoint, ingress, egress = transit_info
        ax1.axvline(midpoint, ls='-.', c='g', alpha=0.5)
        ax1.axvline(ingress, ls='--', c='k', alpha=0.5)
        ax1.axvline(egress, ls='--', c='k', alpha=0.5)

    # Unity
    ax1.axhline(1., ls='--', c='k', alpha=0.5)
    ax1.legend(fontsize=16)

    if 'ylim' in kwargs:
        ax1.set_ylim(kwargs.get('ylim'))

    if 'title' in kwargs:
        ax1.set_title("{}".format(kwargs.get('title')), fontsize=18, y=1.02)

    # Residuals Plot #
    if model_flux is not None:
        ax2 = fig.add_subplot(gs[1])

        residual = y - model_flux
        ax2.plot(residual, color=color, ls='', marker='o',
                 label='Model {:.04f}'.format(residual.std()))

        ax2.axhline(0, ls='--', alpha=0.5)
        ax2.set_title('Model residual (σ={:.02%})'.format(residual.std()))

        if transit_info is not None:
            midpoint, ingress, egress = transit_info
            ax2.axvline(midpoint, ls='-.', c='g', alpha=0.5)
            ax2.axvline(ingress, ls='--', c='k', alpha=0.5)
            ax2.axvline(egress, ls='--', c='k', alpha=0.5)

    fig.tight_layout()

    return fig

