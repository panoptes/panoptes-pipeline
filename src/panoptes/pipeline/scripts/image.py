import os
import re
from pathlib import Path

import numpy as np
import pandas
import pandas as pd
import typer

from astropy import convolution
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.wcs import WCS
from google.cloud import firestore, storage
from photutils import segmentation
from photutils.utils import calc_total_error
from pydantic import BaseModel, BaseSettings
from loguru import logger

from panoptes.pipeline.settings import PipelineParams
from panoptes.pipeline.utils import metadata, sources
from panoptes.pipeline.utils.gcp.bigquery import get_bq_clients
from panoptes.pipeline.utils.gcp.storage import upload_dir
from panoptes.pipeline.utils.metadata import ImageStatus, get_firestore_refs, ObservationStatus, \
    record_metadata
from panoptes.utils.images import fits as fits_utils, bayer
from panoptes.utils.serializers import to_json, from_json

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import papermill as pm


class FileSettings(BaseModel):
    reduced_filename: Path = 'image.fits'
    extras_filename: Path = 'extras.fits'
    metadata_filename: Path = 'metadata.json'
    sources_filename: Path = 'sources.parquet'


class Settings(BaseSettings):
    params: PipelineParams = PipelineParams()
    files: FileSettings = FileSettings()
    compress_fits: bool = True
    output_dir: Path


OUTPUT_BUCKET = os.getenv('OUTPUT_BUCKET', 'panoptes-images-processed')
firestore_db = firestore.Client()
storage_client = storage.Client()

logger.remove()
app = typer.Typer()


@app.command()
def process_notebook(fits_path: str,
                     output_dir: Path,
                     input_notebook: Path = 'ProcessFITS.ipynb',
                     upload: bool = False
                     ):
    typer.secho('Starting image processing')
    output_dir.mkdir(parents=True, exist_ok=True)
    image_settings = Settings(output_dir=output_dir)
    processed_bucket = storage_client.get_bucket(OUTPUT_BUCKET)

    typer.secho(f'Checking if got a fits file at {fits_path}')
    if re.search(r'\d{8}T\d{6}\.fits.*?$', str(fits_path)) is None:
        raise RuntimeError(f'Need a FITS file, got {fits_path}')
    typer.secho(f'Starting processing for {fits_path} in {image_settings.output_dir!r}')

    # Check and update status.
    _, seq_ref, image_doc_ref = get_firestore_refs(fits_path, firestore_db=firestore_db)
    try:
        image_status = image_doc_ref.get(['status']).to_dict()['status']
    except Exception:
        image_status = ImageStatus.UNKNOWN.name

    if ImageStatus[image_status] > ImageStatus.CALIBRATING:
        print(f'Skipping image with status of {ImageStatus[image_status].name}')
        raise FileExistsError(f'Already processed {fits_path}')
    else:
        image_doc_ref.set(dict(status=ImageStatus.CALIBRATING.name), merge=True)
        seq_ref.set(dict(status=ObservationStatus.CREATED.name), merge=True)

    # Run process.
    out_notebook = f'{image_settings.output_dir}/processing-image.ipynb'
    typer.secho(f'Starting {input_notebook} processing')
    try:
        pm.execute_notebook(str(input_notebook),
                            str(out_notebook),
                            parameters=dict(
                                fits_path=str(fits_path),
                                output_dir=str(image_settings.output_dir),
                                image_settings=image_settings.json()
                            ),
                            progress_bar=False,
                            )
    except Exception as e:
        typer.secho(f'Problem processing notebook: {e!r}', color='yellow')
        image_doc_ref.set(dict(status=ImageStatus.ERROR.name), merge=True)
    else:
        with (Path(image_settings.output_dir) / 'metadata.json').open() as f:
            metadata = from_json(f.read())

        full_image_id = metadata['image']['uid'].replace('_', '/')

        firestore_id = record_metadata(fits_path, metadata, firestore_db=firestore_db)
        print(f'Recorded metadata in firestore id={firestore_id}')

        # Upload any assets to storage bucket.
        if upload:
            upload_dir(output_dir, prefix=f'{full_image_id}/', bucket=processed_bucket)
    finally:
        typer.secho(f'Finished processing for {fits_path} in {image_settings.output_dir!r}')


def save_fits(filename, data_list, header, force_new=False):
    hdul = fits.HDUList()
    for name, d in data_list.items():
        hdu = fits.ImageHDU(d, header=header)
        hdu.name = name.upper()
        hdul.append(hdu)

    hdul.writeto(filename, overwrite=force_new)
    typer.secho(f'Saved {len(data_list)} dataset(s) to {filename}')


def get_metadata(header: fits.Header, matched_sources: pandas.DataFrame,
                 settings: Settings) -> dict:
    num_sources = len(matched_sources)
    typer.secho(f'Total sources {num_sources}')
    fwhm_mean, fwhm_median, fwhm_std = sigma_clipped_stats(matched_sources.photutils_fwhm)

    # Puts metadata into better structures.
    metadata_headers = extract_metadata(header)
    metadata_headers['image']['sources'] = dict(num_detected=num_sources,
                                                photutils_fwhm_median=fwhm_median,
                                                photutils_fwhm_mean=fwhm_mean,
                                                photutils_fwhm_std=fwhm_std,
                                                )
    metadata_headers['image']['status'] = ImageStatus.MATCHED.name
    # TODO get rid of encoding loop.
    metadata_headers['image']['params'] = from_json(settings.params.json())

    typer.secho(f'Saving metadata to json file {settings.files.metadata_filename}')
    to_json(metadata_headers, filename=str(settings.files.metadata_filename))
    typer.secho(f'Saved metadata to {settings.files.metadata_filename}.')

    return metadata_headers


def extract_metadata(header):
    typer.echo(f'Removing bad FITS headers (comments, history)')
    # Clear out bad headers.
    header.remove('COMMENT', ignore_missing=True, remove_all=True)
    header.remove('HISTORY', ignore_missing=True, remove_all=True)
    bad_headers = [h for h in header.keys() if h.startswith('_')]
    map(header.pop, bad_headers)

    typer.echo(f'Getting metadata from FITS headers')
    metadata_headers = metadata.extract_metadata(header)

    return metadata_headers


def match_sources(detected_sources: pandas.DataFrame, solved_wcs0: WCS, settings: Settings,
                  image_edge: int = 10) -> pandas.DataFrame:
    typer.secho(f'Matching {len(detected_sources)} sources to wcs.')
    catalog_filename = settings.params.catalog.catalog_filename
    if catalog_filename and catalog_filename.exists():
        typer.secho(f'Using catalog from {settings.params.catalog.catalog_filename}')
        catalog_sources = pd.read_parquet(settings.params.catalog.catalog_filename)
    else:
        typer.secho(f'Getting catalog sources from bigquery for WCS')
        # BQ client.
        bq_client, bqstorage_client = get_bq_clients()
        vmag_limits = settings.params.catalog.vmag_limits
        catalog_sources = sources.get_stars_from_wcs(solved_wcs0,
                                                     bq_client=bq_client,
                                                     bqstorage_client=bqstorage_client,
                                                     vmag_min=vmag_limits[0],
                                                     vmag_max=vmag_limits[1],
                                                     )
    typer.secho(f'Matching sources to catalog for {len(detected_sources)} sources')
    matched_sources = sources.get_catalog_match(detected_sources,
                                                wcs=solved_wcs0,
                                                catalog_stars=catalog_sources,
                                                ra_column='photutils_sky_centroid_ra',
                                                dec_column='photutils_sky_centroid_dec',
                                                max_separation_arcsec=settings.params.catalog.max_separation_arcsec
                                                )
    # Drop matches near border
    typer.secho(f'Filtering sources near within {image_edge} pixels of '
                f'{settings.params.camera.image_width}x{settings.params.camera.image_height}')
    matched_sources = matched_sources.query(
        f'catalog_wcs_x_int > {image_edge} and '
        f'catalog_wcs_x_int < {settings.params.camera.image_width - image_edge} and '
        f'catalog_wcs_y_int > {image_edge} and '
        f'catalog_wcs_y_int < {settings.params.camera.image_height - image_edge}'
    ).copy()
    typer.secho(f'Found {len(matched_sources)} matching sources')

    # There should not be too many duplicates at this point and they are returned in order
    # of catalog separation, so we take the first.
    duplicates = matched_sources.duplicated('picid', keep='first')
    typer.secho(f'Found {len(matched_sources[duplicates])} duplicate sources')

    # Mark which ones were duplicated.
    matched_sources.loc[:, 'catalog_match_duplicate'] = False
    dupes = matched_sources.picid.isin(matched_sources[duplicates].picid)
    matched_sources.loc[dupes, 'catalog_match_duplicate'] = True

    # Filter out duplicates.
    matched_sources = matched_sources.loc[~duplicates].copy()
    typer.secho(f'Found {len(matched_sources)} matching sources after removing duplicates')

    # Add some binned fields that we can use for table partitioning.
    matched_sources['catalog_dec_bin'] = matched_sources.catalog_dec.astype('int')
    matched_sources['catalog_ra_bin'] = matched_sources.catalog_ra.astype('int')

    # Precompute some columns.
    matched_sources[
        'catalog_gaia_bg_excess'] = matched_sources.catalog_gaiabp - matched_sources.catalog_gaiamag
    matched_sources[
        'catalog_gaia_br_excess'] = matched_sources.catalog_gaiabp - matched_sources.catalog_gaiarp
    matched_sources[
        'catalog_gaia_rg_excess'] = matched_sources.catalog_gaiarp - matched_sources.catalog_gaiamag

    return matched_sources


def detect_sources(solved_wcs0, reduced_data, combined_bg_data, combined_bg_residual_data,
                   settings: Settings):
    typer.secho('Detecting sources in image')
    threshold = (settings.params.catalog.detection_threshold * combined_bg_residual_data)
    kernel = convolution.Gaussian2DKernel(3 * gaussian_fwhm_to_sigma)
    kernel.normalize()
    image_segments = segmentation.detect_sources(reduced_data,
                                                 threshold,
                                                 npixels=settings.params.catalog.num_detect_pixels,
                                                 filter_kernel=kernel,
                                                 mask=reduced_data.mask
                                                 )
    typer.secho(f'De-blending image segments')
    deblended_segments = segmentation.deblend_sources(reduced_data,
                                                      image_segments,
                                                      npixels=settings.params.catalog.num_detect_pixels,
                                                      filter_kernel=kernel,
                                                      nlevels=32,
                                                      contrast=0.01)
    typer.secho(
        f'Calculating total error for data using gain={settings.params.camera.effective_gain}')
    error = calc_total_error(reduced_data, combined_bg_residual_data,
                             settings.params.camera.effective_gain)
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
    typer.secho('Building source catalog for deblended_segments')
    detected_catalog = segmentation.SourceCatalog(reduced_data,
                                                  deblended_segments,
                                                  background=combined_bg_data,
                                                  error=error,
                                                  mask=reduced_data.mask,
                                                  wcs=solved_wcs0,
                                                  localbkg_width=settings.params.catalog.localbkg_width_pixels)
    source_cols = sorted(detected_catalog.default_columns + table_cols)
    detected_sources = detected_catalog.to_table(columns=source_cols).to_pandas().dropna()
    # Clean up some column names.
    detected_sources = detected_sources.rename(columns=lambda x: f'photutils_{x}')
    detected_sources = detected_sources.rename(columns={
        'photutils_sky_centroid.ra': 'photutils_sky_centroid_ra',
        'photutils_sky_centroid.dec': 'photutils_sky_centroid_dec',
    })
    return detected_sources


def plate_solve(settings: Settings, filename=None):
    filename = filename or settings.files.reduced_filename
    typer.secho(f'Plate solving {filename}')
    solved_headers = fits_utils.get_solve_field(str(filename),
                                                skip_solved=False,
                                                timeout=300)
    solved_path = solved_headers.pop('solved_fits_file')
    typer.secho(f'Solving completed successfully for {solved_path}')
    solved_wcs0 = WCS(solved_headers)
    return solved_wcs0


def subtract_background(data, settings: Settings):
    # Get RGB background data.
    rgb_background = bayer.get_rgb_background(data=data,
                                              return_separate=True,
                                              box_size=settings.params.background.box_size,
                                              filter_size=settings.params.background.filter_size,
                                              )
    combined_bg_data = list()
    combined_bg_residual_data = list()
    for color, bg in zip(bayer.RGB, rgb_background):
        color_data = np.ma.array(data=bg.background, mask=bg.mask)
        color_residual_data = np.ma.array(data=bg.background_rms, mask=bg.mask)

        combined_bg_data.append(color_data)
        combined_bg_residual_data.append(color_residual_data)
    combined_bg_data = np.ma.array(combined_bg_data).filled(0).sum(0)
    combined_bg_residual_data = np.ma.array(combined_bg_residual_data).filled(0).sum(0)
    reduced_data = data - combined_bg_data

    return combined_bg_data, combined_bg_residual_data, reduced_data


def mask_outliers(data, settings: Settings):
    # Mask min and max outliers.
    # data = np.ma.masked_less_equal(data, 0.)
    data = np.ma.masked_greater_equal(data, settings.params.camera.saturation)
    return data


def subtract_bias(raw_data, settings: Settings):
    # Bias subtract.
    data = raw_data - settings.params.camera.zero_bias
    return data


if __name__ == '__main__':
    app()
