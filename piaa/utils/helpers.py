
import numpy as np

from matplotlib import pyplot as plt

from astropy.table import Table
from astropy.visualization import LogStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from astropy import units as u

from scipy.sparse.linalg import lsqr

from pocs.utils import current_time
from pong.utils.metadb import get_cursor


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

    ax.imshow(data, norm=norm, cmap=get_palette(), origin='lower')

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


def normalize_cube(cube):
    return (cube.T / cube.sum(1)).T


def get_stamp_sq_diff(d0, d1):
    return ((d0 - d1)**2).sum()


def get_ideal_full_coeffs(stamp_collection, damp=1, func=lsqr, verbose=False):

    num_frames = stamp_collection.shape[1]
    num_pixels = stamp_collection.shape[2]

    target_frames = stamp_collection[0].flatten()
    refs_frames = stamp_collection[1:].reshape(-1, num_frames * num_pixels).T

    if verbose:
        print("Target other shape: {}".format(target_frames.shape))
        print("Refs other shape: {}".format(refs_frames.shape))

    coeffs = func(refs_frames, target_frames, damp)

    return coeffs


def get_ideal_full_psc(stamp_collection, coeffs, **kwargs):

    refs = stamp_collection[1:]

    created_frame = (refs.T * coeffs).sum(2).T

    return created_frame
