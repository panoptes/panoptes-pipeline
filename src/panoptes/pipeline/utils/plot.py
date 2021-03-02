from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from astropy.visualization import simple_norm
from panoptes.utils.images import bayer
from panoptes.utils.images import plot as plot_utils


def plot_stamp(picid,
               data,
               metadata,
               frame_idx=None,
               show_mean=False,
               show_all=False,
               cmap=None,
               stretch='sqrt',
               ):
    cmap = cmap or plot_utils.get_palette('viridis')

    picid = metadata.picid.iloc[0]

    fig = Figure()
    FigureCanvas(fig)
    fig.set_figheight(4)
    fig.set_figwidth(4)

    nrows = 1
    ncols = 1
    ax = fig.add_subplot(nrows, ncols, 1)

    if frame_idx:
        stamp = data.iloc[frame_idx]
    else:
        stamp = data

    # Get the frame bounds on full image.
    y0, y1, x0, x1 = metadata.filter(regex='stamp').iloc[frame_idx]

    # Get peak location on stamp.
    x_peak = metadata.catalog_wcs_x_int.iloc[frame_idx] - x0
    y_peak = metadata.catalog_wcs_y_int.iloc[frame_idx] - y0

    # Plot stamp.
    norm = simple_norm(stamp, stretch)
    im0 = ax.imshow(stamp, norm=norm, cmap=cmap, origin='lower')
    plot_utils.add_colorbar(im0)

    # Mean location
    if show_mean:
        ax.scatter(metadata.catalog_wcs_x_mean.astype('int') - x0,
                   metadata.catalog_wcs_y_mean.astype('int') - y0,
                   marker='x',
                   color='lightgreen',
                   edgecolors='red',
                   s=250,
                   label='Catalog - mean position')

    if show_all:
        ax.scatter(metadata.catalog_wcs_x_int - x0,
                   metadata.catalog_wcs_y_int - y0,
                   marker='x',
                   color='orange',
                   edgecolors='orange',
                   s=100,
                   label='Catalog - other frames')

    # Star catalog location for current frame
    ax.scatter(x_peak, y_peak, marker='*', color='yellow', edgecolors='black', s=200,
               label='Catalog - current frame')

    plot_utils.add_pixel_grid(ax,
                              grid_height=data.shape[0],
                              grid_width=data.shape[1],
                              show_superpixel=True,
                              major_alpha=0.3, minor_alpha=0.0, )
    ax.set_xticklabels([t for t in range(int(x0), int(x1) + 2, 2)], rotation=45)
    ax.set_yticklabels([t for t in range(int(y0), int(y1) + 2, 2)])

    ax.legend(loc=3)

    ax.set_title(f'PICID: {picid} Frame: {frame_idx} / {len(metadata)}')
    return fig


def plot_background(rgb_bg_data, title=None):
    """ Plot the RGB backgrounds from `Background2d` objects.

    Args:
        rgb_bg_data (list[photutils.Background2D]): The RGB background data as
            returned by calling `panoptes.utils.images.bayer.get_rgb_background`
            with `return_separate=True`.
        title (str): The title for the plot, default None.

    """

    nrows = 2
    ncols = 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    fig.set_facecolor('white')

    for color in bayer.RGB:
        d0 = rgb_bg_data[color]
        ax0 = axes[0][color]
        ax1 = axes[1][color]

        ax0.set_title(f'{color.name.title()} (med {d0.background_median:.02f} ADU)')
        im = ax0.imshow(d0.background, cmap=f'{color.name.title()}s_r', origin='lower')
        plot_utils.add_colorbar(im)

        ax1.set_title(f'{color.name.title()} rms (med {d0.background_rms_median:.02f} ADU)')
        im = ax1.imshow(d0.background_rms, cmap=f'{color.name.title()}s_r', origin='lower')
        plot_utils.add_colorbar(im)

        ax0.set_axis_off()
        ax1.set_axis_off()

    if title:
        fig.suptitle(title)

    fig.set_size_inches(11, 5)
    return fig