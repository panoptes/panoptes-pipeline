import numpy as np
import pandas as pd
from astropy import stats
from astropy.timeseries import TimeSeries
from matplotlib import pyplot as plt
from astropy.visualization import simple_norm
from matplotlib.figure import Figure
from panoptes.utils.images import bayer, plot
from panoptes.utils.images import plot as plot_utils
import seaborn as sb
from matplotlib import dates as mdates


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


def plot_stamp(data,
               metadata,
               frame_idx=None,
               norm_data=None,
               show_mean=False,
               show_all=False,
               cmap=None,
               stretch='linear',
               title=None,
               mask_alpha=0.25
               ):
    cmap = cmap or plot_utils.get_palette()

    fig, ax = plt.subplots()

    if frame_idx:
        try:
            stamp = data.iloc[frame_idx].to_numpy()
        except AttributeError:
            stamp = data[frame_idx]
    else:
        stamp = data

    stamp_size = stamp.shape

    # Get the frame bounds on full image.
    y0, y1, x0, x1 = metadata.filter(regex='stamp').iloc[frame_idx or 0]

    # Get peak location on stamp.
    x_peak = metadata.catalog_wcs_x_int.iloc[frame_idx or 0] - x0
    y_peak = metadata.catalog_wcs_y_int.iloc[frame_idx or 0] - y0

    # Plot stamp.
    norm_data = norm_data if norm_data is not None else stamp
    norm = simple_norm(norm_data, stretch, min_cut=norm_data.min(), max_cut=norm_data.max() + 50)

    if hasattr(stamp, 'mask'):
        im0 = ax.imshow(stamp, norm=norm, cmap=cmap, origin='lower')
        ax.imshow(np.ma.array(stamp.data, mask=~stamp.mask), norm=norm, cmap=cmap, origin='lower',
                  alpha=mask_alpha)
    else:
        im0 = ax.imshow(stamp, norm=norm, cmap=cmap, origin='lower')

    plot_utils.add_colorbar(im0)

    # Mean location
    if show_mean:
        ax.scatter(metadata.catalog_wcs_x_mean.astype('int') - x0,
                   metadata.catalog_wcs_y_mean.astype('int') - y0,
                   marker='+',
                   color='lightgreen',
                   edgecolors='red',
                   s=250,
                   label='Catalog - mean position')

    if show_all:
        ax.scatter(metadata.catalog_wcs_x_int - x0,
                   metadata.catalog_wcs_y_int - y0,
                   marker='+',
                   color='orange',
                   edgecolors='orange',
                   s=100,
                   label='Catalog - other frames')

    # Star catalog location for current frame
    if frame_idx is not None:
        ax.scatter(x_peak, y_peak, marker='*', color='yellow', edgecolors='black', s=200,
                   label='Catalog - current frame')

    plot_utils.add_pixel_grid(ax,
                              grid_height=stamp_size[0],
                              grid_width=stamp_size[1],
                              show_superpixel=True,
                              major_alpha=0.3, minor_alpha=0.0, )

    # Put pixel locations on labels.
    num_x_labels = len(ax.get_xticks())
    num_y_labels = len(ax.get_yticks())
    x_range = np.linspace(int(x0), int(x1), num_x_labels, dtype=int)
    y_range = np.linspace(int(y0), int(y1), num_y_labels, dtype=int)
    ax.set_xticklabels([t for t in x_range], rotation=45)
    ax.set_yticklabels([t for t in y_range])

    ax.set_xlabel('X pixel coordinate')
    ax.set_ylabel('Y pixel coordinate')

    ax.legend(loc=3)

    if frame_idx is not None:
        ax.set_title(f'Frame: {frame_idx} / {len(metadata)}')

    fig.suptitle(title)
    fig.set_size_inches(14, 6)
    return fig


def wcs_plot(wcs):
    fig = Figure()
    ax = fig.add_subplot(projection=wcs)

    # Make sure WCS shows up.
    ra, dec = ax.coords
    dec.set_ticklabel_position('bt')
    ra.set_ticklabel_position('lr')

    ra.set_major_formatter('d.d')
    dec.set_major_formatter('d.d')
    ra.grid(color='orange', ls='dotted')
    dec.grid(color='orange', ls='dotted')

    ax.set_ylabel('RA (J2000)')
    ax.set_xlabel('Declination (J2000)')

    return ax


def plot_raw_bg_overlay(data, rgb_background, title=None, wcs=None, size=(18, 12)):
    if wcs:
        ax = wcs_plot(wcs)
        fig = ax.figure
    else:
        fig = Figure()
        ax = fig.add_subplot()
    fig.set_size_inches(*size)

    ax.imshow(data, origin='lower', norm=simple_norm(data, 'log', min_cut=0), cmap='Greys')
    ax.grid(False)
    rgb_background.plot_meshes(axes=ax, outlines=True, alpha=0.3, marker='', color='red')

    if title is not None:
        ax.set_title(title)

    height, width = data.shape
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    return fig


def plot_stellar_location(data, title=None, wcs=None):
    ax = wcs_plot(wcs)
    sb.scatterplot(data=data,
                   x='catalog_wcs_x_int',
                   y='catalog_wcs_y_int',
                   hue='catalog_vmag',
                   size='catalog_vmag',
                   sizes=(200, 5),
                   marker='*',
                   edgecolor='black',
                   linewidth=0.2,
                   ax=ax
                   )

    if title:
        ax.figure.suptitle(title, y=0.95)

    ax.figure.set_size_inches(18, 12)

    return ax.figure


def plot_bg_overlay(data, rgb_background, title=None, wcs=None, size=(18, 12)):
    if wcs:
        ax = wcs_plot(wcs)
        fig = ax.figure
    else:
        fig = Figure()
        ax = fig.add_subplot()

    fig.set_size_inches(*size)

    im = ax.imshow(data, origin='lower', cmap='Greys_r', norm=simple_norm(data, 'linear'))
    rgb_background.plot_meshes(axes=ax, outlines=True, alpha=0.1, marker='', color='red')
    plot.add_colorbar(im)

    if title is not None:
        ax.set_title(title)

    ax.grid(False)

    height, width = data.shape
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    return fig


def plot_distribution(data, col, name=None):
    fig = Figure()
    ax = fig.add_subplot()
    sb.histplot(data[col], ax=ax)
    ax.set_title(name or str(col))
    return fig


def filter_plot(data, col, sequence_id):
    fig = Figure()
    ax = fig.add_subplot()

    data[col].plot(ax=ax, marker='.', label='Valid')
    data.query(f'mask_{col}==True')[col].plot(ax=ax, marker='o', color='r', ls='',
                                              label=f'Filtered {col}')

    ax.legend()
    ax.set_xlabel('Time [UTC]')
    ax.set_title(f'Filtered {col} on {sequence_id}')

    fig.set_size_inches(8, 4)
    return fig


def image_simple(d0, title=None, output_file=None, savefig_kwargs=None):
    fig = Figure()
    fig.set_size_inches(18, 12)
    ax = fig.subplots()

    ax.imshow(d0,
              origin='lower',
              norm=simple_norm(d0, stretch='sqrt', min_percent=10.50, max_percent=98.),
              cmap='Greys_r')

    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if title:
        ax.set_title(title)

    if output_file:
        savefig_kwargs = savefig_kwargs or dict(bbox_inches='tight')
        fig.savefig(output_file, **savefig_kwargs)

    return fig


def plot_stamp_comparision(target_df, target_data, comp_data, frame_idx=0, title='',
                           cmap='inferno', stretch='linear'):
    fig = Figure()

    axes = fig.subplots(ncols=3, sharex=True, sharey=True)

    y0, y1, x0, x1 = target_df.filter(regex='stamp').iloc[frame_idx]

    x_peak = target_df.catalog_wcs_x_int.iloc[frame_idx] - x0
    y_peak = target_df.catalog_wcs_y_int.iloc[frame_idx] - y0

    norm = simple_norm(comp_data,
                       stretch,
                       min_cut=comp_data[frame_idx].min(),
                       max_cut=comp_data[frame_idx].max()
                       )

    ax = axes[0]
    im0 = ax.imshow(target_data[frame_idx], cmap=cmap, origin='lower', norm=norm)
    ax.set_title('Scaled target')
    plot_utils.add_colorbar(im0)
    ax.scatter(x_peak, y_peak, marker='*', color='yellow', edgecolors='black', s=200,
               label='Catalog position')

    ax = axes[1]
    im1 = ax.imshow(comp_data[frame_idx], cmap=cmap, origin='lower', norm=norm)
    ax.set_title('Comparison')
    plot_utils.add_colorbar(im1)
    ax.scatter(x_peak, y_peak, marker='*', color='yellow', edgecolors='black', s=200,
               label='Catalog position')

    target_comp_diff0 = target_data - comp_data

    ax = axes[2]
    im2 = ax.imshow(target_comp_diff0[frame_idx], cmap=cmap, origin='lower')
    ax.scatter(x_peak, y_peak, marker='*', color='yellow', edgecolors='black', s=200,
               label='Catalog position')
    plot_utils.add_colorbar(im2)
    ax.set_title('Target-comp residual')

    num_frames, stamp_height, stamp_width = target_data.shape

    for ax in axes:
        # Put pixel locations on labels.
        num_x_labels = len(ax.get_xticks())
        num_y_labels = len(ax.get_yticks())
        x_range = np.linspace(int(x0), int(x1), num_x_labels, dtype=int)
        y_range = np.linspace(int(y0), int(y1), num_y_labels, dtype=int)
        ax.set_xticklabels([t for t in x_range], rotation=45)
        ax.set_yticklabels([t for t in y_range])
        ax.legend(loc=3)
        plot_utils.add_pixel_grid(ax,
                                  stamp_height,
                                  stamp_width,
                                  major_alpha=0.3, minor_alpha=0,
                                  show_superpixel=True)

    fig.set_size_inches(12, 5)
    fig.set_dpi(100)
    fig.suptitle(f'{title}\nFrame {frame_idx}/{num_frames - 1}', y=1.03)

    return fig


def show_raw_aperture_lc(target_df, target_data, aperture, bin_time=30, title=''):
    num_frames, stamp_height, stamp_width = target_data.shape
    target_rgb_data = bayer.get_rgb_data(target_data)
    target_masked_rgb = np.ma.array(target_rgb_data, mask=aperture)

    target_rgb_aperture_raw_df = pd.DataFrame(
        target_masked_rgb.reshape(3, num_frames, -1).sum(2).T).rename(
        columns={c.value: c.name.lower() for c in bayer.RGB})

    target_rgb_aperture_raw_df.index = target_df.index

    target_rgb_aperture_raw_df = target_rgb_aperture_raw_df / target_rgb_aperture_raw_df.mean(0)

    t0_tdf = target_rgb_aperture_raw_df.reset_index().melt(id_vars=['time'], var_name='color')

    t1_tdf = t0_tdf.set_index(['time']).groupby('color').resample(f'{bin_time}T',
                                                                  level='time',
                                                                  label='right',
                                                                  closed='right'
                                                                  ).mean().reset_index()

    raw_std0 = t0_tdf.groupby('color')['value'].std()
    raw_std1 = t1_tdf.groupby('color')['value'].std()

    fig = Figure()
    ax = fig.add_subplot()

    for color in bayer.RGB:
        raw_color_data = t0_tdf[t0_tdf.color == color.name.lower()]
        color_initial = color.name.lower()[0]

        ax.plot(raw_color_data.time, raw_color_data.value,
                marker='.', ms=7, ls='--', alpha=0.25, color=f'{color.name.lower()[0]}',
                label=f'{color_initial} $\sigma = {raw_color_data.value.std():.03f}$')

    for color in bayer.RGB:
        color_initial = color.name.lower()[0]
        binned_color_data = t1_tdf[t1_tdf.color == color.name.lower()]
        ax.plot(binned_color_data.time, binned_color_data.value,
                marker='o', ms=7, ls='--', alpha=0.95, color=f'{color.name.lower()[0]}',
                label=f'{color_initial} binned $\sigma = {binned_color_data.value.std():.03f}$')

    ax.legend()
    ax.set_title('Raw differential RGB photometry')
    ax.figure.suptitle(title)
    ax.figure.set_size_inches(12, 5)

    return fig


def plot_separate_rgb(target_df, target_rgb_lc, comp_rgb_lc, title=''):
    fig = Figure()
    axes = fig.subplots(nrows=3, sharex=True)

    for color in bayer.RGB:
        axes[color].plot(target_df.index.values,
                         comp_rgb_lc[color],
                         color=f'{color.name.lower()[0]}',
                         label=f'Comparison',
                         marker='.')
        axes[color].plot(target_df.index.values,
                         target_rgb_lc[color],
                         color=f'{color.name.lower()[0]}',
                         ls='--',
                         marker='.',
                         label=f'Target')
        axes[color].hlines(1, target_df.index[0], target_df.index[-1], ls='--', alpha=0.5)
        axes[color].legend()

    fig.suptitle(f'Raw relative photometry {title}')
    fig.set_size_inches(12, 5)

    return fig


def make_lc_ts(df, index):
    lc_df = pd.DataFrame(df).T
    lc_df.columns = ['r', 'g', 'b']
    lc_df.index = index

    lc_ts = TimeSeries.from_pandas(lc_df, index)

    return lc_ts


def plot_lightcurve(target_df, diff_rgb_lc, bin_time=30, title=''):
    image_times = target_df.index
    diff_df = make_lc_ts(diff_rgb_lc, image_times).resample(f'{bin_time}T', label='right',
                                                            closed='right').mean()

    fig = Figure()
    ax0 = fig.add_subplot()

    for color in bayer.RGB:
        color_initial = color.name.lower()[0]
        # Raw diff
        ax0.plot(image_times,
                 diff_rgb_lc[color],
                 color=color_initial,
                 alpha=0.25,
                 ls='',
                 marker='.',
                 label=f'{color_initial}    $\sigma = {diff_rgb_lc[color].std():.03f}$',
                 )

    for color in bayer.RGB:
        color_initial = color.name.lower()[0]
        # Time-binned diff
        ax0.plot(diff_df[color_initial],
                 color=color_initial,
                 marker='o',
                 #              lw=1,
                 #              ls='--',
                 label=f'{color_initial} binned $\sigma = {diff_df[color_initial].std():.03f}$'
                 )

    ax0.set_title(f'Differential Photometry - {bin_time} min binning')
    ax0.hlines(1.01, image_times[0], image_times[-1], ls='--', alpha=0.35)
    ax0.hlines(1, image_times[0], image_times[-1], ls='--', alpha=0.65)
    ax0.hlines(0.99, image_times[0], image_times[-1], ls='--', alpha=0.35)

    ax0.legend()

    #     ax0.set_ylim([0.9, 1.1])
    ax0.set_xlabel('Time [UTC]')
    ax0.set_ylabel('Relative flux')

    fig.suptitle(f'{title}')
    fig.set_size_inches(12, 5)

    return fig


def plot_references_location(target_df, refs_df, coeff_df, title='', cmap='inferno', imagew=5208,
                             imageh=3476):
    scatter_df = refs_df.sort_values('score').groupby('picid').mean()
    scatter_df['rank_order'] = scatter_df.score.rank(ascending=True)
    scatter_df = scatter_df.merge(coeff_df, on='picid')

    fig = Figure()
    fig.set_size_inches(12, 8)
    ax = fig.add_subplot()

    sb.scatterplot(
        data=scatter_df[scatter_df.coeffs != 0],
        x='catalog_wcs_x_int',
        y='catalog_wcs_y_int',
        color='red',
        size='rank_order',
        sizes=(200, 100),
        marker='x',
        label='Non-zero coef ref',
        legend=False,
        ax=ax
    )

    sb.scatterplot(
        data=scatter_df,
        x='catalog_wcs_x_int',
        y='catalog_wcs_y_int',
        hue='catalog_vmag',
        palette='viridis_r',
        cmap=cmap,
        size='rank_order',
        sizes=(100, 10),
        marker='o',
        label='References',
        legend=True,
        ax=ax
    )

    sb.scatterplot(
        data=target_df,
        x='catalog_wcs_x_int',
        y='catalog_wcs_y_int',
        marker='*',
        size='catalog_vmag',
        sizes=(500, 500),
        color='red',
        label='Target',
        ax=ax,
        legend=False
    )

    ax.set_xlim([0, imagew])
    ax.set_ylim([0, imageh])
    ax.set_xlabel('Catalog position [pixel]')
    ax.set_ylabel('Catalog position [pixel]')

    ax.set_title(f'Image coordinates showing selected stars {title}')
    # ax.figure.suptitle(f'Image coordinates for $l={num_refs}$ references')
    ax.legend()

    return fig


def plot_ref_scores(top_refs_df):
    fig = Figure()
    fig.set_size_inches(8, 5)
    ax = fig.add_subplot()
    ax.errorbar(range(len(top_refs_df)),
                top_refs_df.score,
                yerr=[np.zeros_like(top_refs_df.coeffs), top_refs_df.coeffs],
                marker='.', ecolor='r', label='Top References w/ coeff value')
    ax.set_title(f'Similarity score and relative coeff value for top references')
    ax.set_xlabel('Similarity index')
    ax.set_ylabel('Similarity score')
    ax.legend()
    return fig


def plot_target_drift(target_df, title='', pixel_scale=10.3):
    fig = Figure()
    fig.set_size_inches(8, 4)
    ax = fig.add_subplot()
    ax.plot(target_df.index, target_df.catalog_wcs_x_drift, marker='o', label='X')
    ax.plot(target_df.index, target_df.catalog_wcs_y_drift, marker='o', label='Y')
    ax.set_xlabel('Time [UTC]')
    ax.set_ylabel('Pixels')

    # Draw 1 pixel peak-to-peak.
    ax.axhline(1., ls='--', color='k', alpha=0.4)
    ax.axhline(0., ls='--', color='k', alpha=0.5)
    ax.axhline(-1., ls='--', color='k', alpha=0.4)

    def pix2arc(y):
        return y * pixel_scale

    def arc2pix(y):
        return y / pixel_scale

    secax = ax.secondary_yaxis('right', functions=(pix2arc, arc2pix))
    secax.set_ylabel('Arcsecs')

    ax.set_ylim([-6, 6])

    ax.set_title(title)
    fig.suptitle('Drift of catalog center pixel drift')

    return fig


def plot_bin_lightcurve(unbinned_ts, binned_ts, binned_error_ts=None, title=''):
    fig = Figure()
    fig.set_size_inches(12, 5)
    ax = fig.subplots()

    for c in bayer.RGB:
        color = c.name.lower()[0]

        # c_mean, c_med, c_std = stats.sigma_clipped_stats(unbinned_ts[color])
        # bc_mean, bc_med, bc_std = stats.sigma_clipped_stats(binned_ts[color])
        c_std = np.nanstd(unbinned_ts[color])
        bc_std = np.nanstd(binned_ts[color])

        label = f'{color}             $\sigma = {c_std:.03f}$'
        ax.plot(unbinned_ts.time.to_datetime(), unbinned_ts[color], color=color, marker='.', ls='',
                alpha=0.5,
                label=label)

        bin_label = f'{color} binned $\sigma = {bc_std:.03f}$'
        ax.errorbar(binned_ts.time_bin_start.to_datetime(), binned_ts[color],
                    binned_error_ts[color], color=color, marker='o', label=bin_label)

        image_times = unbinned_ts.time.to_datetime()
        ax.hlines(1.01, image_times[0], image_times[-1], ls='--', alpha=0.35, color='grey')
        ax.hlines(1, image_times[0], image_times[-1], ls='--', alpha=0.65, color='grey')
        ax.hlines(0.99, image_times[0], image_times[-1], ls='--', alpha=0.35, color='grey')
        #     ax.set_ylim([0.9, 1.1])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel('Time [UTC]')
        ax.set_ylabel('Relative flux')
        ax.legend()

    fig.suptitle(title, y=0.98)
    return fig
