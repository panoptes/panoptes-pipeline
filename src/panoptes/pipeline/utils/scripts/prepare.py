import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import typer
from astropy import convolution
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS

from panoptes.pipeline.utils import metadata, sources
from panoptes.pipeline.utils.gcp.bigquery import get_bq_clients
from panoptes.pipeline.utils.metadata import ImageStatus
from panoptes.utils.images import fits as fits_utils, bayer
from panoptes.utils.serializers import to_json
from photutils import segmentation
from photutils.utils import calc_total_error

from loguru import logger

logger.remove()

app = typer.Typer()


@app.command()
def main(
        url: str,
        output_dir: Path = 'output',
        camera_bias: float = 2048.,
        box_size: Tuple[int, int] = (79, 84),
        filter_size: Tuple[int, int] = (3, 3),
        stamp_size: Tuple[int, int] = (10, 10),
        saturation: float = 11535.0,  # ADU after bias subtraction.
        vmag_min: float = 7,
        vmag_max: float = 12,
        numcont: int = 5,
        localbkg_width: int = 2,
        detection_threshold: float = 5.0,
        num_detect_pixels: int = 4,
        effective_gain: float = 1.5,
        max_catalog_separation: int = 50,
        use_firestore: bool = False,
        use_bigquery: bool = False,
        **kwargs
):
    if re.search(r'\d{8}T\d{6}\.fits[.fz]+$', url) is None:
        raise RuntimeError(f'Need a FITS file, got {url}')

    typer.echo(f'Starting processing for {url} in {output_dir!r}')
    bq_client, bqstorage_client = get_bq_clients()

    output_dir = Path(output_dir)

    if output_dir.exists() is False:
        output_dir.mkdir(exist_ok=True)

    reduced_filename = output_dir / 'calibrated.fits'
    background_filename = output_dir / 'background.fits'
    residual_filename = output_dir / 'background-residual.fits'
    metadata_json_path = output_dir / 'metadata.json'
    matched_path = output_dir / 'matched-sources.csv'

    # Load the image.
    typer.echo(f'Getting data')
    raw_data, header = fits_utils.getdata(url, header=True)

    # Clear out bad headers.
    header.remove('COMMENT', ignore_missing=True, remove_all=True)
    header.remove('HISTORY', ignore_missing=True, remove_all=True)
    bad_headers = [h for h in header.keys() if h.startswith('_')]
    map(header.pop, bad_headers)

    # Bias subtract.
    data = raw_data - camera_bias

    # Mask min and max outliers.
    data = np.ma.masked_less_equal(data, 0.)
    data = np.ma.masked_greater_equal(data, saturation)

    # Get RGB background data.
    typer.echo(f'Getting background for the RGB channels')
    rgb_background = bayer.get_rgb_background(data=data,
                                              mask=data.mask,
                                              return_separate=True,
                                              box_size=box_size,
                                              filter_size=filter_size,
                                              )

    # Combine the RGB background data.
    combined_bg_data = np.ma.array([np.ma.array(data=bg.background, mask=bg.mask)
                                    for bg
                                    in rgb_background]).sum(0).filled(0).astype(np.float32)

    # Also combine the RGB RMS data.
    combined_rms_bg_data = np.ma.array([np.ma.array(data=bg.background_rms, mask=bg.mask)
                                        for bg
                                        in rgb_background]).sum(0).filled(0).astype(np.float32)

    reduced_data = (data - combined_bg_data).data

    # Save reduced data and background.
    hdu0 = fits.PrimaryHDU(reduced_data, header=header)
    hdu0.scale('float32')
    fits.HDUList(hdu0).writeto(reduced_filename)
    typer.echo(f'Saved {reduced_filename}')

    hdu1 = fits.PrimaryHDU(combined_bg_data, header=header)
    hdu1.scale('float32')
    fits.HDUList(hdu1).writeto(background_filename)
    typer.echo(f'Saved {background_filename}')

    hdu2 = fits.PrimaryHDU(combined_rms_bg_data, header=header)
    hdu2.scale('float32')
    fits.HDUList(hdu2).writeto(residual_filename)
    typer.echo(f'Saved {residual_filename}')

    # Plate solve reduced data.
    # TODO Make get_solve_field accept raw data or open file.
    typer.echo(f'Plate solving {reduced_filename}')
    solved_headers = fits_utils.get_solve_field(str(reduced_filename),
                                                skip_solved=False,
                                                timeout=300)
    solved_path = solved_headers.pop('solved_fits_file')
    typer.echo(f'Solving completed successfully for {solved_path}')

    typer.echo(f'Getting stars from catalog')
    solved_wcs0 = WCS(solved_headers)
    catalog_sources = sources.get_stars_from_wcs(solved_wcs0,
                                                 bq_client=bq_client,
                                                 bqstorage_client=bqstorage_client,
                                                 vmag_min=vmag_min,
                                                 vmag_max=vmag_max,
                                                 numcont=numcont,
                                                 )

    typer.echo('Detecting sources in image')
    threshold = (detection_threshold * combined_rms_bg_data)
    kernel = convolution.Gaussian2DKernel(2 * gaussian_fwhm_to_sigma)
    kernel.normalize()
    image_segments = segmentation.detect_sources(reduced_data, threshold, npixels=num_detect_pixels,
                                                 filter_kernel=kernel)
    typer.echo(f'De-blending image segments')
    deblended_segments = segmentation.deblend_sources(reduced_data, image_segments,
                                                      npixels=num_detect_pixels,
                                                      filter_kernel=kernel, nlevels=32,
                                                      contrast=0.01)

    typer.echo(f'Calculating total error for data using gain={effective_gain}')
    error = calc_total_error(reduced_data, combined_rms_bg_data, effective_gain)

    table_cols = [
        'background_mean',
        'cxx', 'cxy', 'cyy',
        'fwhm',
        'kron_radius',
        'perimeter'
    ]
    typer.echo('Building source catalog for deblended_segments')
    source_catalog = segmentation.SourceCatalog(data,
                                                deblended_segments,
                                                background=combined_bg_data,
                                                error=error,
                                                wcs=solved_wcs0,
                                                localbkg_width=localbkg_width)
    source_cols = source_catalog.default_columns + table_cols
    detected_sources = source_catalog.to_table(columns=source_cols).to_pandas().dropna()
    detected_sources = detected_sources.rename(columns=lambda x: f'photutils_{x}')

    typer.echo(f'Matching sources to catalog for {len(detected_sources)} sources')
    matched_sources = sources.get_catalog_match(detected_sources,
                                                wcs=solved_wcs0,
                                                catalog_stars=catalog_sources,
                                                ra_column='photutils_sky_centroid.ra',
                                                dec_column='photutils_sky_centroid.dec',
                                                max_separation_arcsec=max_catalog_separation
                                                )

    # Drop matches near border
    typer.echo(f'Filtering sources near edges')
    image_width, image_height = reduced_data.shape
    matched_sources = matched_sources.query(
        'catalog_wcs_x_int > 10 and '
        f'catalog_wcs_x_int < {image_width - 10} and '
        'catalog_wcs_y_int > 10 and '
        f'catalog_wcs_x_int < {image_height - 10}'
    )
    typer.echo(f'Found {len(matched_sources)} matching sources')

    # Get the xy positions according to the catalog and the wcs.
    stamp_positions = sources.get_xy_positions(solved_wcs0, matched_sources)

    # Get a stamp for each source.
    stamp_positions = stamp_positions.apply(
        lambda row: bayer.get_stamp_slice(row.catalog_wcs_x_int,
                                          row.catalog_wcs_y_int,
                                          stamp_size=stamp_size,
                                          as_slices=False,
                                          ), axis=1,
        result_type='expand')

    stamp_positions.rename(columns={0: 'stamp_y_min',
                                    1: 'stamp_y_max',
                                    2: 'stamp_x_min',
                                    3: 'stamp_x_max'}, inplace=True)

    matched_sources = matched_sources.merge(stamp_positions,
                                            left_index=True,
                                            right_index=True)

    num_sources = len(matched_sources)
    typer.echo(f'Got positions for {num_sources}')

    # Puts metadata into better structures.
    metadata_headers = metadata.extract_metadata(header)
    metadata_headers['image']['matched_sources'] = num_sources


    to_json(metadata_headers, filename=str(metadata_json_path))
    typer.echo(f'Saved metadata to {metadata_json_path}.')

    # Add metadata to matched sources (via json normalization).
    metadata_series = pd.json_normalize(metadata_headers, sep='_').iloc[0]
    matched_sources = matched_sources.assign(**metadata_series)
    matched_sources.set_index(['picid']).to_csv(matched_path, index=True)

    # TODO send to bigquery.

    if use_firestore:
        try:
            typer.echo(f'Saving metadata to firestore')
            image_id = metadata.record_metadata(url,
                                                metadata=metadata_headers,
                                                current_state=ImageStatus.MATCHED)
            typer.echo(f'Saved metadata to firestore with id={image_id}')
        except Exception as e:
            typer.secho(f'Error recording metadata: {e!r}', fg=typer.colors.YELLOW)

    if use_bigquery:
        typer.echo('Pretend upload to bigquery here.')

    return metadata.ObservationPathInfo.from_fits_header(header).get_full_id(sep='/')


if __name__ == '__main__':
    app()
