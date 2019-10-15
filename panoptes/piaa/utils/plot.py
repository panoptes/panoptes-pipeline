import os
import numpy as np
import pandas as pd
from copy import copy
from warnings import warn
from collections import defaultdict
from itertools import chain

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import ticker
from cycler import cycler as cy
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from shapely.geometry import box
from shapely.ops import cascaded_union

from astropy.coordinates import Angle
from astropy.modeling import models, fitting
from astropy import units as u
from astropy.visualization import LogStretch, ImageNormalize, LinearStretch, MinMaxInterval
from astropy.stats import sigma_clip

from panoptes.piaa.utils import helpers


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
    styles = defaultdict(lambda: next(finite_cy_iter))

    return styles


def show_stamps(psc0, psc1,
                frame_idx=0,
                aperture_info=None,
                show_rgb_aperture=True,
                stamp_size=10,
                stretch=None,
                save_name=None,
                show_max=False,
                show_pixel_grid=False,
                cmap='viridis',
                bias_level=2048,
                fig=None,
                **kwargs):

    nrows = 1

    # Two stamps and residual
    ncols = 3

    if show_rgb_aperture:
        ncols += 1

    if fig is None:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_figheight(8)
        fig.set_figwidth(12)
    else:
        axes = fig.axes

    # Get our stamp index
    s0 = psc0[frame_idx]
    s1 = psc1[frame_idx]

    # Get aperture info index
    frame_aperture = None
    if aperture_info is not None:
        aperture_idx = aperture_info.index.levels[0][frame_idx]
        frame_aperture = aperture_info.loc[aperture_idx]

    # Control the stretch
    stretch_method = LinearStretch()
    if stretch == 'log':
        stretch_method = LogStretch()

    norm = ImageNormalize(s1, vmin=bias_level, vmax=s1.max(), stretch=stretch_method)

    # Get axes
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    # Target stamp
    im = ax1.imshow(s0, cmap=get_palette(cmap=cmap), norm=norm)
    ax1.set_title('Target', fontsize=16)

    # Target Colorbar
    # https://stackoverflow.com/questions/13310594/positioning-the-colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.new_horizontal(size="5%", pad=0.05, pack_start=False)
    fig.add_axes(cax)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # Comparison
    im = ax2.imshow(s1, cmap=get_palette(cmap=cmap), norm=norm)
    ax2.set_title('Comparison', fontsize=16)

    # Comparison Colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.new_horizontal(size="5%", pad=0.05, pack_start=False)
    fig.add_axes(cax)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")

    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # Residual
    residual = (s0 - s1) / s1
    im = ax3.imshow(residual, cmap=get_palette(cmap=cmap), norm=ImageNormalize(
        residual, interval=MinMaxInterval(), stretch=LinearStretch()))
    ax3.set_title(f'Residual', fontsize=16)  # Replaced below with aperture residual

    # Residual Colorbar
    divider = make_axes_locatable(ax3)
    cax = divider.new_horizontal(size="5%", pad=0.05, pack_start=False)
    fig.add_axes(cax)
    cbar = fig.colorbar(im,
                        cax=cax,
                        orientation="vertical",
                        #ticks=[residual.min(), 0, residual.max()]
                        )
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # Show apertures
    if frame_aperture is not None:
        # Make the shapely-based aperture
        aperture_pixels = make_shapely_aperture(aperture_info, aperture_idx)
        # TODO: Sometimes holes are appearing.
        full_aperture = cascaded_union([x for x in chain(*aperture_pixels.values())])

        # Get the plotting positions.
        # If there are holes in aperture we get a MultiPolygon
        # and need to handle with a loop. Need to figure out how
        # to handle holes better.
        try:
            xs, ys = full_aperture.exterior.xy

            # Plot aperture mask on target, comparison, and residual.
            ax1.fill(xs, ys, fc='none', ec='orange', lw=3)
            ax2.fill(xs, ys, fc='none', ec='orange', lw=3)
            ax3.fill(xs, ys, fc='none', ec='orange', lw=3)
        except AttributeError:
            for poly in full_aperture:
                xs, ys = poly.exterior.xy

                # Plot aperture mask on target, comparison, and residual.
                ax1.fill(xs, ys, fc='none', ec='orange', lw=3)
                ax2.fill(xs, ys, fc='none', ec='orange', lw=3)
                ax3.fill(xs, ys, fc='none', ec='orange', lw=3)

        # Set the residual title with the std inside the aperture.
        aperture_mask = make_aperture_mask(aperture_info, frame_idx)
        residual_aperture = np.ma.array(data=residual, mask=aperture_mask)

        residual_std = residual_aperture.std()
        ax3.set_title(f'Residual {residual_std:.02f}%', fontsize=16)

        if show_rgb_aperture:
            ax4 = axes[3]

            # Show a checkerboard for bayer (just greyscale)
            bayer = np.ones_like(s0)
            bayer[1::2, 0::2] = 0.1  # Red
            bayer[1::2, 1::2] = 1  # Green
            bayer[0::2, 0::2] = 1  # Green
            bayer[0::2, 1::2] = 0.1  # Blue
            im = ax4.imshow(bayer, alpha=0.17, cmap='Greys')

            # We want the facecolor to be transparent but not the edge
            # so we add transparency directly to facecolor rather than
            # using the normal `alpha` option.
            alpha_value = 0.75
            color_lookup = {
                'r': (1, 0, 0, alpha_value),
                'g': (0, 1, 0, alpha_value),
                'b': (0, 0, 1, alpha_value),
            }

            # Plot individual pixels of the aperture in their appropriate color.
            for color, box_list in aperture_pixels.items():
                for i, b0 in enumerate(box_list):
                    xs, ys = b0.exterior.xy
                    bayer = np.ones((10, 10))
                    ax4.fill(xs, ys, fc=color_lookup[color], ec='k', lw=3)

            add_pixel_grid(ax4, stamp_size, stamp_size, show_superpixel=True)

            ax4.set_title(f'RGB Pattern', fontsize=16)
            ax4.set_yticklabels([])
            ax4.set_xticklabels([])
            ax4.grid(False)

            # Aperture colorbar
            # Add a blank colorbar so formatting is same
            # Todo keep sizes but get rid of colorbar
            divider = make_axes_locatable(ax4)
            cax = divider.new_horizontal(size="5%", pad=0.05, pack_start=False)
            fig.add_axes(cax)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            cbar.ax.set_xticklabels([])
            cbar.ax.set_yticklabels([])

    # Turn off tick labels
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])

    # Turn off grids
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    fig.subplots_adjust(wspace=0.3)

    if save_name:
        try:
            fig.savefig(save_name)
        except Exception as e:
            warn("Can't save figure: {}".format(e))

    return fig


def make_aperture_mask(aperture_info, frame_idx=0, stamp_side=10):
    aperture_idx = aperture_info.index.levels[0][frame_idx]
    mask = np.ones((stamp_side, stamp_side)).astype(bool)

    for color, i in zip('rgb', range(3)):
        for coords in aperture_info.loc[aperture_idx, color].aperture_pixels:
            x = coords[0]
            y = coords[1]
            mask[x, y] = False

    return mask


def make_shapely_aperture(aperture_info, aperture_idx):
    """ Make an aperture for each of the RGB channels """
    pixel_boxes = defaultdict(list)
    for color, i in zip('rgb', range(3)):
        for coords in aperture_info.loc[aperture_idx, color].aperture_pixels:
            x = coords[1]
            y = coords[0]
            b0 = box(x - 0.5, y - 0.5, x + 0.5, y + 0.5)
            pixel_boxes[color].append(b0)

    return pixel_boxes


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
    ax.set_xticks([])
    ax.set_yticks([])

    line = ax.imshow(d0[0])
    ax.set_title(f'Frame 0')

    def animate(i):
        line.set_data(d0[i])  # update the data
        ax.set_title(f'Frame {i:03d}')
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
                im = ax2.imshow(all_channels, vmin=all_channels.min(),
                                vmax=all_channels.max(), origin='lower')

                divider = make_axes_locatable(ax2)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

                plt.setp(ax2.get_xticklabels(), visible=False)
                plt.setp(ax2.get_yticklabels(), visible=False)

            # If first row, set column title
            # if row_num == 0:
            ax.set_title(c_lookup[col_num])
            if col_num == 2:
                ax2.set_title('All')

            # If first column, show frame index to left
            # if col_num == 0:
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


def plot_lightcurve_old(x,
                        y,
                        model_flux=None,
                        use_imag=False,
                        transit_info=None,
                        color='k',
                        **kwargs):
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

    fig.set_size_inches(12, 5)
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


def plot_lightcurve_combined(lc1,
                             time_bin=20,  # Minutes
                             base_model_flux=None,
                             transit_datetimes=None,
                             title=None,
                             colors='rgb',
                             offset_delta=0.
                             ):
    # Setup figure
    fig = Figure()
    FigureCanvas(fig)
    fig.set_size_inches(14, 7)
    fig.set_facecolor('white')

    grid_size = (9, 9)

    # Axis for light curve
    gs = GridSpec(*grid_size, hspace=2)

    # Main plot
    spec1 = gs.new_subplotspec((0, 0), colspan=7, rowspan=7)
    lc_ax = fig.add_subplot(spec1)

    # Residual
    spec2 = gs.new_subplotspec((7, 0), colspan=7, rowspan=2)
    res_scatter_ax = fig.add_subplot(spec2)
    res_scatter_ax.set_ylim([-0.05, 0.05])

    # Residual histos
    res_histo_axes = list()
    for i in range(3):
        spec = GridSpec(*grid_size).new_subplotspec((i * 3, 7), colspan=2, rowspan=3)
        res_ax = fig.add_subplot(spec)

        res_ax.set_xticks([])
        res_ax.set_ylim([-.05, .05])
        res_ax.set_yticks([-.025, 0, .025])
        res_ax.yaxis.tick_right()
        res_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))

        res_histo_axes.append(res_ax)

    offset = 0  # offset

    for i, color in enumerate(colors):

        # Get the normalized flux for each channel
        color_data = lc1.loc[lc1.color == color].copy()

        # Model flux
        if base_model_flux is None:
            base_model = np.ones_like(color_data.flux)
        else:
            # Mask the sigma clipped frames
            base_model = base_model_flux.copy()
        color_data['model'] = base_model

        # Get the differential flux and error
        f0 = color_data.flux
        f0_err = color_data.flux_err
        f0_index = color_data.index
        m0 = color_data.model

        # Flux + offset
        flux = f0 + offset

        # Residual
        residual = flux - base_model

        # Build dataframe for differntial flux
        flux_df = pd.DataFrame({'flux': flux,
                                'flux_err': f0_err,
                                'model': m0,
                                'residual': residual
                                }, index=f0_index,
                               ).dropna()

        # Start plotting.

        # Plot target flux.
        flux_df.flux.plot(yerr=flux_df.flux_err,
                          marker='o', ls='', alpha=0.15, color=color,
                          ax=lc_ax,
                          rot=0,  # Don't rotate date labels
                          legend=False,
                          )

        if time_bin is not None:
            # Time-binned
            binned_flux_df = flux_df.resample(f'{time_bin}T').apply({
                'flux': np.median,
                'flux_err': lambda x: np.sum(x**2)
            })

            # Plot time-binned target flux.
            binned_flux_df.flux.plot(yerr=binned_flux_df.flux_err,
                                     ax=lc_ax,
                                     rot=0,
                                     marker='o', ms=8,
                                     color=color,
                                     ls='',
                                     label=f'Time-binned - {time_bin}min',
                                     legend=False,
                                     )

        # Plot model flux.
        flux_df.model.plot(ax=lc_ax,
                           ls='-',
                           color=color,
                           alpha=0.5,
                           lw=3,
                           rot=0,  # Don't rotate date labels
                           label='Model fit',
                           legend=True
                           )

        # Residual scatter
        flux_df.residual.plot(ax=res_scatter_ax, color=color, ls='', marker='o', alpha=0.5)

        # Residual histogram axis
        res_ax = res_histo_axes[i]

        res_ax.hist(flux_df.residual, orientation='horizontal', color=color, alpha=0.5)
        res_ax.axhline(0, ls='--', color='k', alpha=0.25)
        res_ax.set_title(f'σ={flux_df.residual.std():.2%}', y=.82)

        # Add the offset
        offset += offset_delta

    if transit_datetimes is not None:
        midpoint, ingress, egress = transit_datetimes
        lc_ax.axvline(midpoint, ls='-.', c='g', alpha=0.5)
        lc_ax.axvline(ingress, ls='--', c='k', alpha=0.5)
        lc_ax.axvline(egress, ls='--', c='k', alpha=0.5)

    if title is not None:
        fig.suptitle(title, fontsize=18)

    # Better time axis ticks
    half_hour = mdates.MinuteLocator(interval=30)
    h_fmt = mdates.DateFormatter('%H:%M:%S')

    lc_ax.xaxis.set_major_locator(half_hour)
    lc_ax.xaxis.set_major_formatter(h_fmt)

    res_scatter_ax.set_xticks([])

    return fig


def plot_lightcurve(lc1,
                    time_bin=20,  # Minutes
                    base_model_flux=None,
                    transit_datetimes=None,
                    title=None,
                    colors='rgb',
                    offset_delta=0.
                    ):
    # Setup figure
    fig = Figure()
    FigureCanvas(fig)
    fig.set_size_inches(14, 7)
    fig.set_facecolor('white')

    grid_size = (9, 9)

    # Axis for light curve
    gs = GridSpec(*grid_size, hspace=0.1)

    offset = 0  # offset
    # Better time axis ticks
    half_hour = mdates.MinuteLocator(interval=30)
    h_fmt = mdates.DateFormatter('%H:%M:%S')

    for i, color in enumerate(colors):

        # Light curve plot
        spec1 = gs.new_subplotspec((i * 3, 0), colspan=7, rowspan=2)
        lc_ax = fig.add_subplot(spec1)
        lc_ax.set_xticks([])
        lc_ax.set_ylim([0.93, 1.07])

        # Residual
        spec2 = gs.new_subplotspec((i * 3 + 2, 0), colspan=7, rowspan=1)
        res_scatter_ax = fig.add_subplot(spec2)
        res_scatter_ax.set_ylim([-0.05, 0.05])
        res_scatter_ax.set_yticks([-0.025, 0.025])
        if i != 2:
            res_scatter_ax.set_xticks([])
        else:
            res_scatter_ax.xaxis.set_major_locator(half_hour)
            res_scatter_ax.xaxis.set_major_formatter(h_fmt)

        # Residual histos
        spec = GridSpec(*grid_size).new_subplotspec((i * 3, 7), colspan=2, rowspan=3)
        res_ax = fig.add_subplot(spec)
        res_ax.set_xticks([])
        res_ax.set_ylim([-.05, .05])
        res_ax.set_yticks([-.025, 0, .025])
        res_ax.yaxis.tick_right()
        res_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))

        # Get the normalized flux for each channel
        flux_df = lc1.loc[lc1.color == color].copy()

        # Model flux
        if base_model_flux is None:
            flux_df['model'] = np.ones_like(flux_df.flux)
        else:
            flux_df['model'] = base_model_flux

        # Residual
        flux_df['residual'] = flux_df.flux - flux_df.model

        # Start plotting.

        # Plot target flux.
        flux_df.flux.plot(yerr=flux_df.flux_err,
                          marker='o', ls='', alpha=0.15, color=color,
                          ax=lc_ax,
                          rot=0,  # Don't rotate date labels
                          legend=False,
                          )

        if time_bin is not None:
            # Time-binned
            binned_flux_df = flux_df.resample(f'{time_bin}T').apply({
                'flux': np.mean,
                'flux_err': lambda x: np.sum(x**2)
            })

            # Plot time-binned target flux.
            binned_flux_df.plot(yerr=binned_flux_df.flux_err,
                                ax=lc_ax,
                                rot=0,
                                marker='o', ms=8,
                                color=color,
                                ls='',
                                label=f'Time-binned - {time_bin}min',
                                legend=False,
                                )

        # Plot model flux.
        flux_df.model.plot(ax=lc_ax,
                           ls='-',
                           color=color,
                           alpha=0.5,
                           lw=3,
                           rot=0,  # Don't rotate date labels
                           label='Model fit',
                           legend=True
                           )

        # Residual scatter
        flux_df.residual.plot(ax=res_scatter_ax,
                              color=color,
                              ls='',
                              marker='o',
                              rot=0,  # Don't rotate date labels
                              alpha=0.5
                              )

        # Residual histogram
        res_ax.hist(flux_df.residual, orientation='horizontal', color=color, alpha=0.5)
        res_ax.axhline(0, ls='--', color='k', alpha=0.25)
        res_ax.set_title(f'σ={flux_df.residual.std():.2%}', y=.82)

        # Add the offset
        offset += offset_delta

    if transit_datetimes is not None:
        midpoint, ingress, egress = transit_datetimes
        lc_ax.axvline(midpoint, ls='-.', c='g', alpha=0.5)
        lc_ax.axvline(ingress, ls='--', c='k', alpha=0.5)
        lc_ax.axvline(egress, ls='--', c='k', alpha=0.5)

    if title is not None:
        fig.suptitle(title, fontsize=18)

    return fig


def make_sigma_aperture_plot(masked_stamps=None, add_pixel_grid=False):
    cmap_lookup = {
        'r': 'Reds',
        'g': 'Greens',
        'b': 'Blues'
    }

    fig = Figure()
    FigureCanvas(fig)
    fig.set_size_inches(9, 3)

    gs = GridSpec(1, 3, figure=fig)

    full_ax = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    cax1 = fig.add_subplot(gs[2])

    # Show the full data - without mask any of 'rgb' are full data
    full_ax.imshow(masked_stamps['r'].data)

    for i, color in enumerate('rgb'):
        masked_stamp = masked_stamps[color]
        # Mask values below zero for display
        masked_stamp.mask[masked_stamp <= 0] = True

        # Show plots
        ax.imshow(masked_stamp,
                  vmin=0,
                  cmap=cmap_lookup[color], alpha=0.95)
        cax1.bar(i, masked_stamp.sum(),
                 color=color, align='edge',
                 label=f'{masked_stamp.sum():.0f}')
        # Add a dummy bar to get width
        cax1.bar(i + 3, 0)

    full_ax.set_xticks([])
    full_ax.set_yticks([])
    if add_pixel_grid:
        add_pixel_grid(ax, 10, 10,
                       show_axis_labels=False,
                       show_superpixel=True,
                       major_alpha=0.05, minor_alpha=0.025
                       )

    ax.set_facecolor('#bbbbbb')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)

    cax1.legend(fontsize=12)
    cax1.set_ylim([0, 1e5])
    cax1.set_xticks([])
    cax1.set_yticks([])
    cax1.set_xticklabels([])
    cax1.set_yticklabels([])
    cax1.grid(False)

    full_ax.set_title('Full stamp')
    ax.set_title(f'Aperture pixels')
    cax1.set_title(f'Aperture sum')

    # p = Polygon([(0.5, 2.5), (1.5, 2.5)], linewidth=2, edgecolor='black')
    # ax.add_patch(p)

    return fig


def show_stamp_widget(masked_stamps):
    from IPython import display
    from ipywidgets import widgets

    text_wid = widgets.IntText(
        value=0,
        placeholder='Frame number',
        description='Frame number:',
        disabled=False
    )
    slider_wid = widgets.IntSlider(
        value=0,
        min=0,
        max=len(masked_stamps),
        step=1,
        description='Frames:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    widgets.jslink((text_wid, 'value'), (slider_wid, 'value'))

    output_widget = widgets.Output(layout={'height': '250px'})

    def show_stamp(i):
        fig = make_sigma_aperture_plot(masked_stamps[i])
#             fig.set_size_inches(9, 2.5)
        fig.suptitle(f'Frame {i:03d}', fontsize=14, y=1.06)
        fig.axes[1].grid(False)
        fig.axes[1].set_yticklabels([])
        fig.axes[1].set_facecolor('#bbbbbb')
        with output_widget:
            display.clear_output()
            display.display(fig)

    def on_value_change(change):
        frame_idx = change['new']
        show_stamp(frame_idx)

    slider_wid.observe(on_value_change, names='value')

    frame_box = widgets.HBox([output_widget])
    control_box = widgets.HBox([text_wid, slider_wid])  # Controls

    main_box = widgets.VBox([frame_box, control_box])

    display.display(main_box)
    show_stamp(0)


def show_obs_plot_widget(data, display_fn):
    from IPython import display
    from ipywidgets import widgets

    text_wid = widgets.IntText(
        value=0,
        placeholder='Frame number',
        description='Frame number:',
        disabled=False
    )
    slider_wid = widgets.IntSlider(
        value=0,
        min=0,
        max=len(data),
        step=1,
        description='Frames:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    widgets.jslink((text_wid, 'value'), (slider_wid, 'value'))

    output_widget = widgets.Output(layout={'height': '250px'})

    def on_value_change(change):
        frame_idx = change['new']
        display_fn(frame_idx)

    slider_wid.observe(on_value_change, names='value')

    frame_box = widgets.HBox([output_widget])
    control_box = widgets.HBox([text_wid, slider_wid])  # Controls

    main_box = widgets.VBox([frame_box, control_box])

    display.display(main_box)
    display_fn(0)
