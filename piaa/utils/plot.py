from copy import copy

from matplotlib import pyplot as plt

from astropy.visualization import LogStretch, ImageNormalize, LinearStretch
from photutils import RectangularAperture, RectangularAnnulus

from piaa.utils import helpers


def get_palette():
    palette = copy(plt.cm.inferno)
    palette.set_over('w', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('g', 1.0)
    return palette


def show_stamps(
        idx_list=None,
        pscs=None,
        frame_idx=0,
        stamp_size=11,
        aperture_size=4,
        show_residual=False,
        stretch=None,
        **kwargs
):

    midpoint = (stamp_size - 1) / 2
    aperture = RectangularAperture((midpoint, midpoint), w=aperture_size, h=aperture_size, theta=0)
    annulus = RectangularAnnulus((midpoint, midpoint), w_in=aperture_size,
                                 w_out=stamp_size, h_out=stamp_size, theta=0)

    if idx_list is not None:
        pscs = [helpers.get_psc(i, stamp_size=stamp_size, **kwargs) for i in idx_list]
        ncols = len(idx_list)
    else:
        ncols = len(pscs)

    if show_residual:
        ncols += 1

    fig, ax = plt.subplots(nrows=2, ncols=ncols)
    fig.set_figheight(6)
    fig.set_figwidth(12)

    norm = [helpers.normalize(p) for p in pscs]

    s0 = pscs[0][frame_idx]
    n0 = norm[0][frame_idx]

    s1 = pscs[1][frame_idx]
    n1 = norm[1][frame_idx]

    if stretch == 'log':
        stretch = LogStretch()
    else:
        stretch = LinearStretch()

    # Target
    ax1 = ax[0][0]
    im = ax1.imshow(s0.reshape(stamp_size, stamp_size), origin='lower',
                    cmap=get_palette(), norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax1)
    annulus.plot(color='c', lw=2, ls='--', ax=ax1)
    fig.colorbar(im, ax=ax1)

    # Normalized target
    ax2 = ax[1][0]
    im = ax2.imshow(n0.reshape(stamp_size, stamp_size), origin='lower',
                    cmap=get_palette(), norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax2)
    annulus.plot(color='c', lw=2, ls='--', ax=ax2)
    fig.colorbar(im, ax=ax2)
    ax2.set_title('Normalized Stamp')

    # Comparison
    ax1 = ax[0][1]
    im = ax1.imshow(s1.reshape(stamp_size, stamp_size), origin='lower',
                    cmap=get_palette(), norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax1)
    annulus.plot(color='c', lw=2, ls='--', ax=ax1)
    fig.colorbar(im, ax=ax1)

    # Normalized comparison
    ax2 = ax[1][1]
    im = ax2.imshow(n1.reshape(stamp_size, stamp_size), origin='lower',
                    cmap=get_palette(), norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax2)
    annulus.plot(color='c', lw=2, ls='--', ax=ax2)
    fig.colorbar(im, ax=ax2)
    ax2.set_title('Normalized Stamp')

    if show_residual:

        # Residual
        ax1 = ax[0][2]
        im = ax1.imshow((s0 - s1).reshape(stamp_size, stamp_size), origin='lower',
                        cmap=get_palette(), norm=ImageNormalize(stretch=stretch))
        aperture.plot(color='r', lw=4, ax=ax1)
        annulus.plot(color='c', lw=2, ls='--', ax=ax1)
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Stamp Residual - {:.02f}'.format((s0 - s1).sum()))

        # Normalized residual
        ax2 = ax[1][2]
        im = ax2.imshow((n0 - n1).reshape(stamp_size, stamp_size),
                        origin='lower', cmap=get_palette())
        aperture.plot(color='r', lw=4, ax=ax2)
        annulus.plot(color='c', lw=2, ls='--', ax=ax2)
        fig.colorbar(im, ax=ax2)
        ax2.set_title('Normalized Stamp')

    fig.tight_layout()
