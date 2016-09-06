import os

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np
import pandas as pd

from pocs.utils.images import conversions


def make_postage_stamp(data, x, y, radius=3):
    round_x = np.round(x)
    round_y = np.round(y)

    if round_x % 2 == 1:
        round_x = round_x + 1

    if round_y % 2 == 1:
        round_y = round_y + 1

    center_x = int(round_x)
    center_y = int(round_y)

    d1 = data[center_y - radius:center_y + radius, center_x - radius: center_x + radius]
    return d1[::-1]


def get_postage_stamp(fn, ra, dec, radius=3):
    # Solve FITS
    if not os.path.exists(fn.replace('.cr2', '.fits')):
        conversions.process_cr2(fn, solve=True, verbose=False)

    with fits.open(fn.replace('.cr2', '.fits')) as hdu:
        data = hdu[0].data
        header = hdu[0].header

    # Get WCS
    wcs = WCS(header)
    pix = wcs.wcs_world2pix([[ra, dec]], 1)

    stamp = make_postage_stamp(data, pix[0][0], pix[0][1], radius=radius)

    return stamp, header['DATE-OBS']


def write_psc(stamps, name, times=None):
    hdu = fits.PrimaryHDU(np.array(stamps))

    if times is not None:
        for i, t in enumerate(times):
            hdu.header['TIME{:04}'.format(i)] = t.isoformat()

    hdulist = fits.HDUList([hdu])
    hdulist.writeto(name, clobber=True)


def make_masks(data):
    w, h = data.shape

    red_mask = np.array(
        [index[0] % 2 == 0 and index[1] % 2 == 0 for index, i in np.ndenumerate(data)]
    ).reshape(w, h)

    blue_mask = np.array(
        [index[0] % 2 == 1 and index[1] % 2 == 1 for index, i in np.ndenumerate(data)]
    ).reshape(w, h)

    green_mask = np.array(
        [(index[0] % 2 == 0 and index[1] % 2 == 1) or (index[0] % 2 == 1 and index[1] % 2 == 0)
         for index, i in np.ndenumerate(data)]
    ).reshape(w, h)

    return red_mask, green_mask, blue_mask


def make_psc(files, ra, dec, name, radius=3, save_fits=False):
    stamps = list()
    times = list()
    for f in files:
        psc, ts = get_postage_stamp(f, ra, dec, radius=radius)
        stamps.append(psc)
        times.append(ts)

    times = pd.to_datetime(times)

    if save_fits:
        write_psc(stamps, '{}.fits'.format(name), times=times)

    return stamps, times
