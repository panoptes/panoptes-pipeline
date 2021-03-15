from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from astropy.stats import sigma_clip

from panoptes.utils.images import bayer
from panoptes.utils.logging import logger


def get_rectangle_aperture(stamp_size: Tuple[int, int], annulus_width: int = 2):
    """Gets a square aperture.

    Args:
        stamp_size (Tuple[int, int]):  The full aperture stamp size.
        annulus_width (int): The width of the annulus, default 2 pixels.

    Returns:
        npt.ArrayLike: The aperture mask.
    """
    center_aperture = np.ones(stamp_size)

    center_aperture[:annulus_width] = 0
    center_aperture[:, :annulus_width] = 0
    center_aperture[:, -annulus_width:] = 0
    center_aperture[-annulus_width:] = 0

    aperture_mask = ~np.ma.make_mask(center_aperture)

    return aperture_mask


def get_rgb_sigma_clip_aperture(data: npt.ArrayLike, sigma: int = 2, **kwargs):
    """Make a sigma clipper aperture for each of the RGB channels.

    Args:
        data (numpy.typing.ArrayLike): The data to be clipped. This should be an
            array of size [3 x num_frames x stamp_area] or
            [3 x num_frames x stamp_width x stamp_height].
        sigma (int): The sigma to use for clipping.
        kwargs (dict): Keyword options passed to the `astropy.stats.sigma_clip`.

    Returns:
        npt.ArrayLike: The aperture mask.
    """
    if len(data.shape) == 4:
        stamp_size = data.shape[-2:]
        stamp_area = stamp_size[0] * stamp_size[1]
    else:
        stamp_area = data.shape[-1]
        stamp_side = int(np.sqrt(stamp_area))
        stamp_size = (stamp_side, stamp_side)

    # Aperture is size of stamp.
    aperture_mask = np.ones([stamp_area]).reshape(stamp_size)

    for i, color in enumerate(bayer.RGB):
        d0 = data[i].mean(0).reshape(stamp_size)
        clipped_data = sigma_clip(d0.copy(), sigma=sigma)

        # Add clipped data to aperture.
        aperture_mask = np.logical_and(aperture_mask, clipped_data.mask)

    # Negate aperture logic.
    aperture_mask = ~aperture_mask

    return aperture_mask


def make_location_aperture(x: npt.ArrayLike,
                           y: npt.ArrayLike,
                           aperture_size: Tuple[int, int] = (10, 10),
                           dilation_iterations: int = 1,
                           ):
    """Make dilated apertures from the given x and y coordinate locations.

    The `x` and `y` arrays can come from any source. Using PIPELINE metadata the
    mean WCS catalog positions can be used like:

        x_locs = (dataframe.catalog_wcs_x_int - dataframe.stamp_x_min).values
        y_locs = (dataframe.catalog_wcs_y_int - dataframe.stamp_y_min).values
        make_location_aperture(x_locs, y_locs, (10, 10))

    Args:
        x (npt.ArrayLike): The x coordinate locations to dilate.
        y (npt.ArrayLike): The y coordinate locations to dilate.
        aperture_size (Tuple[int, int]): The initial size of the aperture, default (10, 10).
        dilation_iterations (int): The number of binary dilations, default 1.

    Returns:
        npt.ArrayLike: The aperture mask.
    """
    aperture_mask = np.zeros(aperture_size)

    for loc in list(zip(y, x)):
        try:
            aperture_mask[loc] = 1
        except Exception as e:
            logger.warning(f"Bad coordinate location for mask at {x=},{y=} {e!r}")

    if dilation_iterations > 0:
        aperture_mask = ndimage.binary_dilation(aperture_mask, iterations=dilation_iterations)

    # Negate the mask
    aperture_mask = ~np.ma.make_mask(aperture_mask)
    return aperture_mask
