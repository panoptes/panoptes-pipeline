import os

import numpy as np
from astropy.table import Table

from astropy.visualization import LogStretch, ImageNormalize, LinearStretch, PercentileInterval
from astropy.wcs import WCS

from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr

from matplotlib import pyplot as plt

from astropy import units as u

from copy import copy

from pocs.utils import current_time
from pong.utils.metadb import get_cursor

palette = copy(plt.cm.inferno)
palette.set_over('w', 1.0)
palette.set_under('k', 1.0)
palette.set_bad('g', 1.0)


def get_stars_from_footprint(wcs_footprint, **kwargs):
    ra = wcs_footprint[:, 0]
    dec = wcs_footprint[:, 1]

    return get_stars(ra.min(), ra.max(), dec.min(), dec.max(), **kwargs)


def get_stars(ra_min, ra_max, dec_min, dec_max, table='full_catalog', cursor_only=True, verbose=False, *args, **kwargs):
    cur = get_cursor(instance='tess-catalog', db='v6')
    cur.execute('SELECT id, ra, dec, tmag, e_tmag, twomass FROM {} WHERE tmag < 13 AND ra >= %s AND ra <= %s AND dec >= %s AND dec <= %s;'.format(
        table), (ra_min, ra_max, dec_min, dec_max))

    if cursor_only:
        return cur

    d0 = np.array(cur.fetchall())
    if verbose:
        print(d0)
    return Table(data=d0, names=['id', 'ra', 'dec', 'tmag', 'e_tmag', 'twomass'], dtype=['i4', 'f8', 'f8', 'f4', 'f4', 'U26'])


def get_star_info(twomass_id, table='full_catalog', verbose=False):
    cur = get_cursor(instance='tess-catalog', db='v6')

    cur.execute('SELECT * FROM {} WHERE twomass=%s'.format(table), (twomass_id,))
    d0 = np.array(cur.fetchall())
    if verbose:
        print(d0)
    return d0


def make_pretty_from_fits(header, data, figsize=(10, 8), dpi=150, alpha=0.2, pad=3.0, **kwargs):
    wcs = WCS(header)
    data = np.ma.array(data, mask=(data > 12000))

    title = kwargs.get('title', header.get('FIELD', 'Unknown'))
    exp_time = header.get('EXPTIME', 'Unknown')

    filter_type = header.get('FILTER', 'Unknown filter')
    date_time = header.get('DATE-OBS', current_time(pretty=True)).replace('T', ' ', 1)

    percent_value = kwargs.get('normalize_clip_percent', 99.9)

    title = '{} ({}s {}) {}'.format(title, exp_time, filter_type, date_time)
    norm = ImageNormalize(interval=PercentileInterval(percent_value), stretch=LogStretch())

    plt.figure(figsize=figsize, dpi=dpi)

    if wcs.is_celestial:
        ax = plt.subplot(projection=wcs)
        ax.coords.grid(True, color='white', ls='-', alpha=alpha)

        ra_axis = ax.coords['ra']
        dec_axis = ax.coords['dec']

        ra_axis.set_axislabel('Right Ascension')
        dec_axis.set_axislabel('Declination')

        ra_axis.set_major_formatter('hh:mm')
        dec_axis.set_major_formatter('dd:mm')

        ra_axis.set_ticks(spacing=5 * u.arcmin, color='white', exclude_overlapping=True)
        dec_axis.set_ticks(spacing=5 * u.arcmin, color='white', exclude_overlapping=True)

        ra_axis.display_minor_ticks(True)
        dec_axis.display_minor_ticks(True)

        dec_axis.set_minor_frequency(10)
    else:
        ax = plt.subplot()
        ax.grid(True, color='white', ls='-', alpha=alpha)

        ax.set_xlabel('X / pixels')
        ax.set_ylabel('Y / pixels')

    ax.imshow(data, norm=norm, cmap=palette, origin='lower')

    plt.tight_layout(pad=pad)
    plt.title(title)

    new_filename = 'pretty.png'
    plt.savefig(new_filename)
#     plt.show()

    plt.close()


def get_rgb_masks(data):

    rgb_mask_file = 'rgb_masks.npz'
    try:
        return np.load(rgb_mask_file)
    except FileNotFoundError:
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

        green_mask = np.flipud(np.array(
            [(index[0] % 2 == 0 and index[1] % 2 == 1) or (index[0] % 2 == 1 and index[1] % 2 == 0)
             for index, i in np.ndenumerate(data)]
        ).reshape(w, h))

        _rgb_masks = np.array([red_mask, green_mask, blue_mask])

        _rgb_masks.dump(rgb_mask_file)

        return _rgb_masks


def get_all_sum(cube):
    return [get_sum(stamp) for stamp in cube]


def get_sum(stamp, stamp_size=11):
    # Get sums for aperture and annulus
    phot_table = aperture_photometry(stamp.reshape(
        stamp_size, stamp_size), (aperture, annulus), method='subpixel', subpixels=32)

    # Get annulus per pixel (local background)
    bkg_mean = phot_table['aperture_sum_1'] / annulus.area()

    # Get background in aperture
    bkg_sum = aperture.area() * bkg_mean

    # Remove local background
    final_sum = phot_table['aperture_sum_0'] - bkg_sum

    return final_sum[0]


def get_psc(idx=None, ticid=None, aperture_size=None, get_masks=False, stamp_size=11, stamp_dir=None, stamp_cubes=None, verbose=False):
    if idx is not None:
        d0 = np.load(stamp_cubes[idx])

    if ticid is not None:
        d0 = np.load(os.path.join(stamp_dir, '{}.npz'.format(ticid)))

    psc = d0['psc']
    pos = d0['pos']
    if verbose:
        print(pos)

    midpoint = int((stamp_size-1)/2)

    masks = list()
    if get_masks:
        if aperture_size is not None:
            size = aperture_size
        else:
            size = stamp_size
        for color, mask in rgb_masks.items():
            masks.append(
                np.array([Cutout2D(mask, p, size, mode='strict').data.flatten() for p in pos]))
    else:
        if aperture_size is not None:
            psc = np.array([Cutout2D(s.reshape(stamp_size, stamp_size), (midpoint,
                                                                         midpoint), aperture_size, mode='strict').data.flatten() for s in psc])

    if get_masks is False:
        return psc
    else:
        return np.array(masks)


def show_stamps(idx_list=None, pscs=None, frame_idx=0, stamp_size=11, aperture_size=4, show_residual=False, stretch=None, **kwargs):

    midpoint = (stamp_size - 1) / 2
    aperture = RectangularAperture((midpoint, midpoint), w=aperture_size, h=aperture_size, theta=0)
    annulus = RectangularAnnulus((midpoint, midpoint), w_in=aperture_size,
                                 w_out=stamp_size, h_out=stamp_size, theta=0)

    if idx_list is not None:
        pscs = [get_psc(i, stamp_size=stamp_size, **kwargs) for i in idx_list]
        ncols = len(idx_list)
    else:
        ncols = len(pscs)

    if show_residual:
        ncols += 1

    fig, ax = plt.subplots(nrows=2, ncols=ncols)
    fig.set_figheight(6)
    fig.set_figwidth(12)

    norm = [normalize(p) for p in pscs]

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
                    cmap=palette, norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax1)
    annulus.plot(color='c', lw=2, ls='--', ax=ax1)
    fig.colorbar(im, ax=ax1)
    #ax1.set_title('Stamp {:.02f}'.format(get_sum(s0, stamp_size=stamp_size)))

    # Normalized target
    ax2 = ax[1][0]
    im = ax2.imshow(n0.reshape(stamp_size, stamp_size), origin='lower',
                    cmap=palette, norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax2)
    annulus.plot(color='c', lw=2, ls='--', ax=ax2)
    fig.colorbar(im, ax=ax2)
    ax2.set_title('Normalized Stamp')

    # Comparison
    ax1 = ax[0][1]
    im = ax1.imshow(s1.reshape(stamp_size, stamp_size), origin='lower',
                    cmap=palette, norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax1)
    annulus.plot(color='c', lw=2, ls='--', ax=ax1)
    fig.colorbar(im, ax=ax1)
    #ax1.set_title('Stamp {:.02f}'.format(get_sum(s1, stamp_size=stamp_size)))

    # Normalized comparison
    ax2 = ax[1][1]
    im = ax2.imshow(n1.reshape(stamp_size, stamp_size), origin='lower',
                    cmap=palette, norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax2)
    annulus.plot(color='c', lw=2, ls='--', ax=ax2)
    fig.colorbar(im, ax=ax2)
    ax2.set_title('Normalized Stamp')

    if show_residual:

        # Residual
        ax1 = ax[0][2]
        im = ax1.imshow((s0 - s1).reshape(stamp_size, stamp_size), origin='lower',
                        cmap=palette, norm=ImageNormalize(stretch=stretch))
        aperture.plot(color='r', lw=4, ax=ax1)
        annulus.plot(color='c', lw=2, ls='--', ax=ax1)
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Stamp Residual - {:.02f}'.format((s0 - s1).sum()))

        # Normalized residual
        ax2 = ax[1][2]
        im = ax2.imshow((n0 - n1).reshape(stamp_size, stamp_size), origin='lower', cmap=palette)
        aperture.plot(color='r', lw=4, ax=ax2)
        annulus.plot(color='c', lw=2, ls='--', ax=ax2)
        fig.colorbar(im, ax=ax2)
        ax2.set_title('Normalized Stamp')

    fig.tight_layout()


def normalize(cube):
    return (cube.T / cube.sum(1)).T


def get_vary(d0, d1):
    return ((d0 - d1)**2).sum()


def spiral_matrix(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[:, 0][::-1])  # take first row and reverse it
        A = A[:, 1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)


def get_ideal_coeffs(stamp_collection, func=None, verbose=False):
    coeffs = []

    def minimize_func(refs_coeffs, references, targets):
        compare_references = (references * refs_coeffs).sum(0)
#         compare_references = (references.T * refs_coeffs).sum(2).T

#         res = ((targets - compare_references)**2)
        res = ((targets - compare_references)**2)

        return res.sum()

    if func is None:
        func = minimize_func

    num_refs = stamp_collection.shape[0] - 1
    num_frames = stamp_collection.shape[1]
    num_pixels = stamp_collection.shape[2]

    target_frames = stamp_collection[0]
    refs_frames = stamp_collection[1:]

    for frame_index in range(num_frames):

        target_all_but_frame = np.delete(target_frames, frame_index, axis=0)
        refs_all_but_frame = np.delete(refs_frames, frame_index, axis=1)

        try:
            # Try to start from previous frame coeffs
            refs_coeffs = coeffs[-1]
        except IndexError:
            # Otherwise all ones
            refs_coeffs = np.ones(num_pixels)
#             refs_coeffs = np.ones(num_refs)

        # Reshape is basically flattening along all but axis 0
#         refs_all_but_frame = refs_all_but_frame.reshape(-1, -1, refs_coeffs.flatten().shape[0])

        if verbose and frame_index == 0:
            print("Target other shape: {}".format(target_all_but_frame.shape))
            print("Refs other shape: {}".format(refs_all_but_frame.shape))
            print("Source coeffs shape: {}".format(refs_coeffs.shape))

        res = minimize(func, refs_coeffs, args=(refs_all_but_frame, target_all_but_frame))
        coeffs.append(res.x)

    return np.array(coeffs)


def get_ideal_full_coeffs(stamp_collection, damp=1, func=lsqr, verbose=False):

    num_refs = stamp_collection.shape[0] - 1
    num_frames = stamp_collection.shape[1]
    num_pixels = stamp_collection.shape[2]

    target_frames = stamp_collection[0].flatten()
    refs_frames = stamp_collection[1:].reshape(-1, num_frames * num_pixels).T

    if verbose:
        print("Target other shape: {}".format(target_frames.shape))
        print("Refs other shape: {}".format(refs_frames.shape))

    coeffs = func(refs_frames, target_frames, damp)

    return coeffs


def get_ideal_psc(stamp_collection, coeffs, **kwargs):
    num_frames = stamp_collection.shape[1]

    # References we will multiple by the coeffs
    refs = stamp_collection[1:]
    print(refs.shape)
    created_frames = []

    for frame_index in range(num_frames):
        # References for this frame
        refs_frame = refs[:, frame_index]

        created_frame = (refs_frame * coeffs[frame_index]).sum(0).T.flatten()
#         created_frame = (refs_frame.T * coeffs[frame_index]).T.sum(0).flatten()
        created_frames.append(created_frame)

    return np.array(created_frames)


def get_ideal_full_psc(stamp_collection, coeffs, **kwargs):

    num_frames = stamp_collection.shape[1]

    refs = stamp_collection[1:]

    created_frame = (refs.T * coeffs).sum(2).T

    return created_frame
