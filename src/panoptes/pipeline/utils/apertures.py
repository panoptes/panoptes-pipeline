import numpy as np
from scipy import ndimage
from astropy.stats import sigma_clip

from panoptes.utils.images import bayer


def get_rectangle_aperture(stamp_size, annulus_width=2):
    """Gets a square aperture.

    plt.imshow(stamps.get_rectangle_aperture(annulus_width=2))
    """
    center_aperture = np.ones(stamp_size)

    center_aperture[:annulus_width] = 0
    center_aperture[:, :annulus_width] = 0
    center_aperture[:, -annulus_width:] = 0
    center_aperture[-annulus_width:] = 0

    aperture_mask = ~np.ma.make_mask(center_aperture)

    return aperture_mask


def get_sigma_clip_aperture(data, sigma=2):
    """
    plt.imshow(stamps.get_sigma_clip_aperture(stamp_size=stamp_size, sigma=2))
    """
    stamp_area = data.shape[-1]

    # Assume square stamp.
    stamp_size = int(np.sqrt(stamp_area))

    # Aperture is size of stamp.
    aperture_mask = np.ones([stamp_area]).reshape(stamp_size)

    for i, color in enumerate(bayer.RGB):
        d0 = data[i].mean(0).reshape((stamp_size, stamp_size))
        clipped_data = sigma_clip(d0.copy(), sigma=sigma)

        # Add clipped data to aperture.
        aperture_mask = np.logical_and(aperture_mask, clipped_data.mask)

    # Negate aperture logic.
    aperture_mask = ~aperture_mask

    return aperture_mask


def make_location_aperture(dataframe, stamp_size, dilation_iterations=1):
    aperture_mask = np.zeros(stamp_size)

    x_locs = (dataframe.catalog_wcs_x_int - dataframe.stamp_x_min).values
    y_locs = (dataframe.catalog_wcs_y_int - dataframe.stamp_y_min).values

    for loc in list(zip(y_locs, x_locs)):
        try:
            aperture_mask[loc] = 1
        except Exception as e:
            print(e)

    if dilation_iterations > 0:
        aperture_mask = ndimage.binary_dilation(aperture_mask, iterations=dilation_iterations)

    # Negate the mask
    aperture_mask = ~np.ma.make_mask(aperture_mask)

    return aperture_mask
