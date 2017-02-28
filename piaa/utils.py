import os
import shutil
import subprocess

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS

from numba import autojit

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import ipywidgets as widgets

from pocs.utils import error
from pocs.utils import images

from photutils import RectangularAnnulus
from photutils import RectangularAperture


class StampSizeException(Exception):
    pass


@autojit
def get_variance(s0, s1):
    """ Compare one stamp to another and get variance

    Args:
        stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
        frame(int): The frame number we want to compare
        i(int): Index of target PIC
        j(int): Index of PIC we want to compare target to
    """
    return ((s0 - s1)**2).sum()


@autojit
def get_all_variance(stamps, i, normalize=False):
    """ Get all variances for given target

    Args:
        stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
        i(int): Index of target PIC
    """
    num_stars = stamps.shape[0]
    num_frames = stamps.shape[1]

    v = np.zeros((num_stars), dtype=np.float)

    for m in range(num_frames):
        s0 = stamps[i, m]
        if normalize:
            s0 = s0 / s0.sum()

        for j in range(num_stars):
            s1 = stamps[j, m]
            if normalize:
                s1 = s1 / s1.sum()
            v[j] += get_variance(s0, s1)

    return v


@autojit
def get_long(stamps):
    num_stars = stamps.shape[0]

    D = np.zeros((num_stars, num_stars), dtype=np.float)

    for i in range(num_stars):
        for j in range(num_stars):
            if i > j:
                continue
            tmp = stamps[i] - stamps[j]
            tmp *= tmp
            D[i, j] = np.sum(tmp)

    return D


def get_cube(seq_files, point_sources, radius=3, padding=(0, 0, 0, 0), normalize=False, bias_subtract=1024):

    cube = np.ndarray((len(point_sources), len(seq_files), (2*radius)**2))

    # x1 = 4.5
    # y1 = 4.5
    # apertures = RectangularAperture((x1, y1), w=6, h=6, theta=0)
    # annulus = RectangularAnnulus((x1, y1), w_in=6, w_out=10, h_out=10, theta=0)

    for i, f in enumerate(seq_files):
        if i % 10 == 0:
            print(i, end='')
        else:
            print('.', end='')
        with fits.open(f) as hdu:
            if f.endswith('.fz'):
                idx = 1
            else:
                idx = 0

            header = hdu[idx].header
            d0 = hdu[idx].data - bias_subtract
            # Get common background
            d_mean, d_median, d_std = sigma_clipped_stats(d0)
            d0 = d0 - int(d_mean)
            d0[d0 > 1e4] = 0

            wcs = WCS(header)

            # See if image is solved
            if not wcs.is_celestial:
                print('not solved')
                continue

            coords = wcs.all_world2pix(point_sources['ALPHA_J2000'], point_sources['DELTA_J2000'], 1)

            for j, ps in enumerate(point_sources):
                # Get the postage stamp around the coordinates
                c0 = make_postage_stamp(d0, coords[0][j], coords[1][j], radius=radius, padding=padding)
                #c0 = Cutout2D(d0, (get_index(coords[0][j]), get_index(coords[1][j])), (10, 10), wcs=wcs).data - ps['BACKGROUND']

                cube[j, i] = c0.flatten()
                #cube[j, i] = get_rgb(c0).flatten()

                # Find a rough estimate of the background in the annulus
                # bkg_flux = aperture_photometry(c1, annulus)['aperture_sum'][0] / annulus.area()

                # c2 = c1 - bkg_flux

                # # Get the flux for each channel located within the aperture
                # if normalize:
                #     cube[j, i] = (c2 / c2.sum()).flatten()
                # else:
                #     # cube[j, i] = background_subtract(c0)
                #     cube[j, i] = c2.flatten()

            hdu.close()

    return cube


@autojit
def get_rgb(c0):
    r_mask, g_mask, b_mask = make_masks(c0.reshape(6, 6))

    r_d = np.ma.array(c0, mask=~r_mask)
    g_d = np.ma.array(c0, mask=~g_mask)
    b_d = np.ma.array(c0, mask=~b_mask)

    r_median = np.ma.median(r_d)
    g_median = np.ma.median(g_d)
    b_median = np.ma.median(b_d)

    # x1 = 4.5
    # y1 = 4.5
    # # apertures = RectangularAperture((x1, y1), w=6, h=6, theta=0)
    # annulus = RectangularAnnulus((x1, y1), w_in=6, w_out=10, h_out=10, theta=0)

    # # r_flux = aperture_photometry(c0, apertures, mask=~r_mask)['aperture_sum'][0]
    # r_bkg_flux = aperture_photometry(c0, annulus, mask=~r_mask)['aperture_sum'][0] / (annulus.area() / 4)

    # # g_flux = aperture_photometry(c0, apertures, mask=~g_mask)['aperture_sum'][0]
    # g_bkg_flux = aperture_photometry(c0, annulus, mask=~g_mask)['aperture_sum'][0] / (annulus.area() / 2)

    # # b_flux = aperture_photometry(c0, apertures, mask=~b_mask)['aperture_sum'][0]
    # b_bkg_flux = aperture_photometry(c0, annulus, mask=~b_mask)['aperture_sum'][0] / (annulus.area() / 4)

    # # print(r_bkg_flux, g_bkg_flux, b_bkg_flux)

    # r_d = np.ma.array(c0 - r_bkg_flux, mask=~r_mask).flatten()
    # g_d = np.ma.array(c0 - g_bkg_flux, mask=~g_mask).flatten()
    # b_d = np.ma.array(c0 - b_bkg_flux, mask=~b_mask).flatten()

    return (r_d - r_median).filled(0) + (g_d - g_median).filled(0) + (b_d - b_median).filled(0)


def background_subtract(c0, normalize=False):
    # Get the color-channel masks
    annulus = RectangularAnnulus((x1, y1), w_in=6, w_out=10, h_out=10, theta=0)
    r_mask, g_mask, b_mask = make_masks(c0.reshape(10, 10))

    # Compute the sigma clipped stamps for each of the color stamps
    r_mean, r_median, r_std = sigma_clipped_stats(c0, mask=~r_mask)
    g_mean, g_median, g_std = sigma_clipped_stats(c0, mask=~g_mask)
    b_mean, b_median, b_std = sigma_clipped_stats(c0, mask=~b_mask)

    # Get the background subtracted data for each stamp (flattened into 1-D array)
    r_d = np.ma.array(c0 - r_median, mask=~r_mask).flatten()
    g_d = np.ma.array(c0 - g_median, mask=~g_mask).flatten()
    b_d = np.ma.array(c0 - b_median, mask=~b_mask).flatten()

    # Get the flux for each channel located within the aperture
    if normalize:
        r_d /= r_d.sum()
        g_d /= g_d.sum()
        b_d /= b_d.sum()

    return (r_d, g_d, b_d)


def get_point_sources(field_dir, seq_files, image_num=0, sextractor_params=None, force_new=False):
    # Write the sextractor catalog to a file
    source_file = '{}/test{:02d}.cat'.format(field_dir, image_num)

    if not os.path.exists(source_file) or force_new:
        # Build catalog of point sources
        sextractor = shutil.which('sextractor')
        if sextractor is None:
            raise error.InvalidSystemCommand('sextractor not found')

        if sextractor_params is None:
            sextractor_params = [
                '-c', '{}/PIAA/resources/conf_files/sextractor/panoptes.sex'.format(os.getenv('PANDIR')),
                '-CATALOG_NAME', source_file,
            ]

        cmd = [sextractor, *sextractor_params, seq_files[image_num]]
        subprocess.run(cmd)

    # Read catalog
    point_sources = Table.read(source_file, format='ascii.sextractor')

    # Remove the point sources that sextractor has flagged
    if 'FLAGS' in point_sources.keys():
        point_sources = point_sources[point_sources['FLAGS'] == 0]
        point_sources.remove_columns(['FLAGS'])

    # Rename columns
    point_sources.rename_column('X_IMAGE', 'X')
    point_sources.rename_column('Y_IMAGE', 'Y')

    # Filter point sources near edge
    # w, h = data[0].shape
    w, h = (3476, 5208)

    stamp_size = 50

    top = point_sources['Y'] > stamp_size
    bottom = point_sources['Y'] < w - stamp_size
    left = point_sources['X'] > stamp_size
    right = point_sources['X'] < h - stamp_size

    return point_sources[top & bottom & right & left]


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


def compare_psc(pscs, w=6, h=6, with_colorbar=False):

    @widgets.interact(i=widgets.IntSlider(min=0, max=pscs.shape[1] - 1, step=1, description="Frame #"))
    def draw_stamps(i):
        sns.set_style('white')
        fig, axes = plt.subplots(1, pscs.shape[0])

        for num, stamp in enumerate(pscs[:, i]):
            im = axes[num].imshow(stamp.reshape(w, h), cmap="Greys")
            axes[num].set_title('Flux={:.2f}'.format(stamp.sum()))

        if with_colorbar:
            cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            fig.colorbar(im, cax=cax)

        fig.set_size_inches(9, 10)
        plt.tight_layout()


def get_index(x):
    return int(round(x - 1))


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


def get_postage_stamp(fn, ra, dec, radius=3):
    # Solve FITS
    # Radius here is full image to solve
    images.get_solve_field(fn, ra=ra, dec=dec, radius=15)

    with fits.open(fn) as hdu:
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
