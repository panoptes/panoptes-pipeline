from warnings import warn

from matplotlib import pyplot as plt

from skimage.feature import canny
from skimage.transform import hough_circle
from skimage.transform import hough_circle_peaks

import numpy as np

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import WCS

from pocs.utils import images as img_utils


def analyze_polar_rotation(pole_fn):
    img_utils.get_solve_field(pole_fn)

    wcs = WCS(pole_fn)

    pole_cx, pole_cy = wcs.all_world2pix(360, 90, 1)

    return pole_cx, pole_cy


def analyze_ra_rotation(rotate_fn):
    d0 = fits.getdata(rotate_fn)  # - 2048

    # Get center
    position = (d0.shape[1] // 2, d0.shape[0] // 2)
    size = (1500, 1500)
    d1 = Cutout2D(d0, position, size)

    d1.data = d1.data / d1.data.max()

    # Get edges for rotation
    rotate_edges = canny(d1.data, sigma=1.0)

    rotate_hough_radii = np.arange(100, 500, 50)
    rotate_hough_res = hough_circle(rotate_edges, rotate_hough_radii)
    rotate_accums, rotate_cx, rotate_cy, rotate_radii = \
        hough_circle_peaks(rotate_hough_res, rotate_hough_radii, total_num_peaks=1)

    return d1.to_original_position((rotate_cx[-1], rotate_cy[-1]))


def plot_center(pole_fn, rotate_fn, pole_center, rotate_center):

    d0 = fits.getdata(pole_fn) - 2048.
    d1 = fits.getdata(rotate_fn) - 2048.

    d0 /= d0.max()
    d1 /= d1.max()

    pole_cx, pole_cy = pole_center
    rotate_cx, rotate_cy = rotate_center

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 14))

    ax.scatter(rotate_cx, rotate_cy, color='r', marker='x', lw=5)

    norm = ImageNormalize(stretch=SqrtStretch())

    ax.imshow(d0 + d1, cmap='Greys_r', norm=norm, origin='lower')

    if (pole_cy - rotate_cy) > 50:
        ax.arrow(rotate_cx, rotate_cy, pole_cx - rotate_cx, pole_cy -
                 rotate_cy, fc='r', ec='r', width=20, length_includes_head=True)
    else:
        ax.scatter(pole_cx, pole_cy, color='r', marker='x', lw=5)

    return fig
