from typing import Union, Tuple, Optional

import numpy as np
import pandas
from astropy import convolution
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from loguru import logger
from panoptes.pipeline.model import Image
from panoptes.pipeline.settings import BackgroundParams
from panoptes.pipeline.utils import sources
from panoptes.pipeline.utils.sources import get_catalog_match
from panoptes.utils.images import fits as fits_utils, bayer
from photutils.segmentation import SourceCatalog, detect_sources, deblend_sources
from photutils.utils import calc_total_error

logger.disable('panoptes')


def subtract_background(image: Image,
                        background_params: Union[BackgroundParams, dict] = BackgroundParams(),
                        ):
    """Calculate the RGB background for a Bayer array FITS image.

    Args:
        image (Image): The image model.
        background_params (BackgroundParams): The parameters to use for background
            subtraction.
    """
    if isinstance(background_params, dict):
        background_params = BackgroundParams(**background_params)

    data = image.data.data.copy()

    # Mask outliers
    data = np.ma.masked_greater_equal(data, background_params.saturation)
    data = np.ma.masked_less_equal(data, 0.)

    # Bias subtract.
    logger.info(f'Subtracting {background_params.camera_bias=}')
    data -= background_params.camera_bias

    # Get RGB background data.
    rgb_background = bayer.get_rgb_background(data=data,
                                              mask=data.mask,
                                              return_separate=True,
                                              box_size=background_params.box_size,
                                              filter_size=background_params.filter_size,
                                              )

    # Combine the RGB background data.
    combined_bg_data = np.ma.array([np.ma.array(data=bg.background, mask=bg.mask)
                                    for bg
                                    in rgb_background]).sum(0).filled(0).astype(np.float32)

    # Also combine the RGB RMS data.
    combined_rms_bg_data = np.ma.array([np.ma.array(data=bg.background_rms, mask=bg.mask)
                                        for bg
                                        in rgb_background]).sum(0).filled(0).astype(np.float32)

    # Assign various data back to image.
    image.data.data = data
    image.data.background = combined_bg_data
    image.data.background_rms = combined_rms_bg_data

    # Headers to mark processing status.
    image.metadata.extra['image']['calibration'] = dict(
        bias_subtracted=True,
        background_subtracted=True,
        background_params=background_params.dict()
    )

    logger.success(f'Calibrated {image}')


def plate_solve(image: Image):
    """Receives the message and process necessary steps."""
    logger.debug(f"Starting plate-solving for FITS file {image}")
    solved_headers = fits_utils.get_solve_field(image.metadata.filepath,
                                                skip_solved=False,
                                                timeout=300)

    solved_path = solved_headers.pop('solved_fits_file')
    solved_headers['FILENAME'] = solved_path

    image.data.wcs = WCS(solved_headers)
    logger.success(f'Solving completed successfully for {image}')


def match_sources(image: Image,
                  catalog_sources: pandas.DataFrame,
                  localbkg_width: Optional[int] = 2,
                  detection_threshold: float = 5.0,
                  num_detect_pixels: int = 4,
                  effective_gain: float = 1.5
                  ):
    """Look up and catalog match the sources in the image."""

    data = image.data.data - image.data.background

    threshold = (detection_threshold * image.data.background_rms)
    kernel = convolution.Gaussian2DKernel(2 * gaussian_fwhm_to_sigma)
    kernel.normalize()
    logger.info('Detecting sources')
    image_segments = detect_sources(data, threshold, npixels=num_detect_pixels,
                                    filter_kernel=kernel)
    logger.info(f'De-blending image segments')
    deblended_segments = deblend_sources(data, image_segments, npixels=num_detect_pixels,
                                         filter_kernel=kernel, nlevels=32,
                                         contrast=0.01)

    logger.info(f'Calculating total error for data using gain={effective_gain}')
    error = calc_total_error(data, image.data.background_rms, effective_gain)

    table_cols = [
        'background_mean',
        'cxx', 'cxy', 'cyy',
        'fwhm',
        'kron_radius',
        'perimeter'
    ]
    logger.info('Building source catalog for deblended_segments')
    source_catalog = SourceCatalog(data,
                                   deblended_segments,
                                   background=image.data.background,
                                   error=error,
                                   wcs=image.data.wcs,
                                   localbkg_width=localbkg_width)
    source_cols = source_catalog.default_columns + table_cols
    detected_sources = source_catalog.to_table(columns=source_cols).to_pandas().dropna()
    detected_sources = detected_sources.rename(columns=lambda x: f'photutils_{x}')

    logger.info(f'Matching sources to catalog for {len(detected_sources)} sources')
    image.data.sources = get_catalog_match(detected_sources,
                                           wcs=image.data.wcs,
                                           catalog_stars=catalog_sources,
                                           ra_column='photutils_sky_centroid.ra',
                                           dec_column='photutils_sky_centroid.dec',
                                           max_separation_arcsec=None
                                           )

    logger.info(f'Found {len(image.data.sources)} matching sources')


def get_source_stamps(image: Image,
                      stamp_size: Tuple[int, int] = (10, 10)
                      ):
    logger.info(f'Looking up sources for {image} {image.data.wcs}')

    # Get the xy positions according to the catalog and the wcs.
    stamp_positions = sources.get_xy_positions(image.data.wcs, image.data.sources)

    # Get a stamp for each source.
    stamp_positions = stamp_positions.apply(lambda row: bayer.get_stamp_slice(row.catalog_wcs_x_int,
                                                                              row.catalog_wcs_y_int,
                                                                              stamp_size=stamp_size,
                                                                              as_slices=False,
                                                                              ), axis=1,
                                            result_type='expand')

    stamp_positions.rename(columns={0: 'stamp_y_min',
                                    1: 'stamp_y_max',
                                    2: 'stamp_x_min',
                                    3: 'stamp_x_max'}, inplace=True)

    image.data.sources = image.data.sources.merge(stamp_positions,
                                                  left_index=True,
                                                  right_index=True)

    logger.info(f'Got positions for {len(image.data.sources)}')
