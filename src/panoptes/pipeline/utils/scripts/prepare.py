import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import typer
from astropy import convolution
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.wcs import WCS
from google.cloud import bigquery, firestore
from loguru import logger
from panoptes.pipeline.utils import metadata, sources
from panoptes.pipeline.utils.gcp.bigquery import get_bq_clients
from panoptes.pipeline.utils.metadata import ImageStatus
from panoptes.utils.images import fits as fits_utils, bayer
from panoptes.utils.serializers import to_json
from photutils import segmentation
from photutils.utils import calc_total_error

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
        vmag_min: float = 6,
        vmag_max: float = 14,
        numcont: int = 5,
        localbkg_width: int = 2,
        detection_threshold: float = 5.0,
        num_detect_pixels: int = 4,
        effective_gain: float = 1.5,
        max_catalog_separation: int = 50,
        bq_table_id: str = 'panoptes-exp.observations.matched_sources',
        force_new: bool = False,
        **kwargs
):
    processing_id = hashlib.md5(url.encode()).hexdigest()

    def _print(string, **print_kwargs):
        typer.secho(f'{processing_id} {string}', **print_kwargs)

    _print('Starting preparation')

    # Print helper so log messages can be tracked to processing run.
    _print('Setting up print functions')

    _print(f'Checking if got a fits file at {url}')
    if re.search(r'\d{8}T\d{6}\.fits[.fz]+$', url) is None:
        raise RuntimeError(f'Need a FITS file, got {url}')
    _print(f'Starting processing for {url} in {output_dir!r}')

    # BQ client.
    bq_client, bqstorage_client = get_bq_clients()
    firestore_db = firestore.Client()

    # Set up output directory and filenames.
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    reduced_filename = output_dir / 'calibrated.fits'
    background_filename = output_dir / 'background.fits'
    residual_filename = output_dir / 'background-residual.fits'
    metadata_json_path = output_dir / 'metadata.json'
    matched_path = output_dir / 'matched-sources.csv'

    # Load the image.
    _print(f'Getting data')
    raw_data, header = fits_utils.getdata(url, header=True)

    # Puts metadata into better structures.
    _print(f'Getting metadata')
    metadata_headers = metadata.extract_metadata(header)

    # Get the path info.
    path_info = metadata.ObservationPathInfo.from_fits_header(header)

    # Clear out bad headers.
    header.remove('COMMENT', ignore_missing=True, remove_all=True)
    header.remove('HISTORY', ignore_missing=True, remove_all=True)
    bad_headers = [h for h in header.keys() if h.startswith('_')]
    map(header.pop, bad_headers)

    # Record the received data and check for duplicates.
    image_id = None
    try:
        _print(f'Saving metadata to firestore')
        image_id = metadata.record_metadata(url,
                                            metadata=deepcopy(metadata_headers),
                                            current_state=ImageStatus.RECEIVED,
                                            firestore_db=firestore_db,
                                            force_new=force_new
                                            )
        _print(f'Saved metadata to firestore with id={image_id}')
    except FileExistsError:
        if force_new is False:
            _print(f'File has already been processed, skipping', fg=typer.colors.YELLOW)
            raise FileExistsError
        else:
            _print(f'File has been processed previously, but {force_new=} so proceeding.')
    except Exception as e:
        _print(f'Error recording metadata: {e!r}', fg=typer.colors.RED)
        return

    if image_id is None:
        raise RuntimeError('Missing image_id for some reason. Giving up.')

    # Bias subtract.
    data = raw_data - camera_bias

    # Mask min and max outliers.
    data = np.ma.masked_less_equal(data, 0.)
    data = np.ma.masked_greater_equal(data, saturation)

    # Get RGB background data.
    _print(f'Getting background for the RGB channels')
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

    reduced_data_object = (data - combined_bg_data)
    reduced_data = reduced_data_object.data.astype(np.float32)

    # Save reduced data and background.
    hdu0 = fits.PrimaryHDU(reduced_data, header=header)
    hdu0.scale('float32')
    fits.HDUList(hdu0).writeto(reduced_filename)
    _print(f'Saved {reduced_filename}')

    hdu1 = fits.PrimaryHDU(combined_bg_data, header=header)
    hdu1.scale('float32')
    fits.HDUList(hdu1).writeto(background_filename)
    _print(f'Saved {background_filename}')

    hdu2 = fits.PrimaryHDU(combined_rms_bg_data, header=header)
    hdu2.scale('float32')
    fits.HDUList(hdu2).writeto(residual_filename)
    _print(f'Saved {residual_filename}')

    # Plate solve reduced data.
    # TODO Make get_solve_field accept raw data or open file.
    _print(f'Plate solving {reduced_filename}')
    solved_headers = fits_utils.get_solve_field(str(reduced_filename),
                                                skip_solved=False,
                                                timeout=300)
    solved_path = solved_headers.pop('solved_fits_file')
    _print(f'Solving completed successfully for {solved_path}')

    _print(f'Getting stars from catalog')
    solved_wcs0 = WCS(solved_headers)
    # Todo: adjust vmag based on exptime.
    catalog_sources = sources.get_stars_from_wcs(solved_wcs0,
                                                 bq_client=bq_client,
                                                 bqstorage_client=bqstorage_client,
                                                 vmag_min=vmag_min,
                                                 vmag_max=vmag_max,
                                                 numcont=numcont,
                                                 )

    _print('Detecting sources in image')
    threshold = (detection_threshold * combined_rms_bg_data)
    kernel = convolution.Gaussian2DKernel(2 * gaussian_fwhm_to_sigma)
    kernel.normalize()
    image_segments = segmentation.detect_sources(reduced_data,
                                                 threshold,
                                                 npixels=num_detect_pixels,
                                                 filter_kernel=kernel)
    _print(f'De-blending image segments')
    deblended_segments = segmentation.deblend_sources(reduced_data,
                                                      image_segments,
                                                      npixels=num_detect_pixels,
                                                      filter_kernel=kernel,
                                                      nlevels=32,
                                                      contrast=0.01)

    _print(f'Calculating total error for data using gain={effective_gain}')
    error = calc_total_error(reduced_data, combined_rms_bg_data, effective_gain)

    table_cols = [
        'background_mean',
        'background_centroid',
        'background_sum',
        'cxx', 'cxy', 'cyy',
        'eccentricity',
        'equivalent_radius',
        'fwhm',
        'gini',
        'kron_radius',
        'perimeter'
    ]
    _print('Building source catalog for deblended_segments')
    source_catalog = segmentation.SourceCatalog(reduced_data,
                                                deblended_segments,
                                                background=combined_bg_data,
                                                error=error,
                                                mask=reduced_data_object.mask,
                                                wcs=solved_wcs0,
                                                localbkg_width=localbkg_width)
    source_cols = sorted(source_catalog.default_columns + table_cols)
    detected_sources = source_catalog.to_table(columns=source_cols).to_pandas().dropna()

    # Clean up some column names.
    detected_sources = detected_sources.rename(columns=lambda x: f'photutils_{x}')
    detected_sources = detected_sources.rename(columns={
        'photutils_sky_centroid.ra': 'photutils_sky_centroid_ra',
        'photutils_sky_centroid.dec': 'photutils_sky_centroid_dec',
    })

    _print(f'Matching sources to catalog for {len(detected_sources)} sources')
    matched_sources = sources.get_catalog_match(detected_sources,
                                                wcs=solved_wcs0,
                                                catalog_stars=catalog_sources,
                                                ra_column='photutils_sky_centroid_ra',
                                                dec_column='photutils_sky_centroid_dec',
                                                max_separation_arcsec=max_catalog_separation
                                                )

    # Drop matches near border
    _print(f'Filtering sources near edges')
    image_height, image_width = reduced_data.shape
    matched_sources = matched_sources.query(
        'catalog_wcs_x_int > 10 and '
        f'catalog_wcs_x_int < {image_width - 10} and '
        'catalog_wcs_y_int > 10 and '
        f'catalog_wcs_y_int < {image_height - 10}'
    )
    _print(f'Found {len(matched_sources)} matching sources')

    # There should not be too many duplicates at this point and they are returned in order
    # of catalog separation, so we take the first.
    duplicates = matched_sources.duplicated('picid', keep='first')
    _print(f'Found {len(matched_sources[duplicates])} duplicate sources')

    # Mark which ones were duplicated.
    dupes = matched_sources.picid.isin(matched_sources[duplicates].picid)
    matched_sources.loc[:, 'catalog_match_duplicate'] = False
    matched_sources.loc[dupes, 'catalog_match_duplicate'] = True

    # Filter out duplicates.
    matched_sources = matched_sources.loc[~duplicates].copy()

    _print(f'Found {len(matched_sources)} matching sources after removing duplicates')

    # Add some binned fields that we can use for table partitioning.
    matched_sources['catalog_dec_bin'] = matched_sources.catalog_dec.astype('int')
    matched_sources['catalog_ra_bin'] = matched_sources.catalog_ra.astype('int')

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
    _print(f'Got positions for {num_sources}')

    fwhm_mean, fwhm_median, fwhm_std = sigma_clipped_stats(matched_sources.photutils_fwhm)

    # Update some of the firestore record.
    _print(f'Adding firestore record.')
    metadata_headers['image']['num_sources'] = num_sources

    firestore_db.document(image_id).set({
        'num_sources': num_sources,
        'status': ImageStatus.MATCHED.name,
        'fwhm_median': fwhm_median,
        'fwhm_std': fwhm_std,
    }, merge=True)

    _print(f'Saving metadata to json file {metadata_json_path}: {metadata_headers!r}')
    to_json(metadata_headers, filename=str(metadata_json_path))
    _print(f'Saved metadata to {metadata_json_path}.')

    # Add metadata to matched sources (via json normalization).
    metadata_series = pd.json_normalize(metadata_headers, sep='_').iloc[0]
    matched_sources = matched_sources.assign(**metadata_series)

    # Remove some duplicated information.
    columns_to_drop = [
        'sequence_project',
        'sequence_unit_id',
        'sequence_camera_id',
        'image_sequence_id',
        'image_image_camera_id',
    ]
    matched_sources.drop(columns=columns_to_drop, errors='ignore')

    # Write dataframe to csv.
    matched_sources.set_index(['picid']).to_csv(matched_path, index=True)
    _print(f'Matched sources saved to {matched_path}')

    _print(f'Uploading to BigQuery table {bq_table_id}')
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        ignore_unknown_values=True,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    try:
        job = bq_client.load_table_from_dataframe(matched_sources, bq_table_id,
                                                  job_config=job_config)
        job.result()  # Start and wait for the job to complete.
        if job.error_result:
            _print(f'Errors while loading BQ job: {job.error_result!r}', fg=typer.colors.RED)
        else:
            _print(f'Finished uploading {job.output_rows} to BigQuery table {bq_table_id}',
                   fg=typer.colors.GREEN)
    except Exception as e:
        _print(f'Error inserting into BigQuery: {e!r}')

    # Return a path-like string.
    return path_info.get_full_id(sep='/')


if __name__ == '__main__':
    app()
