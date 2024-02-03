from pathlib import Path

import numpy as np
import pandas
import pandas as pd
from astropy import convolution
from astropy.coordinates import SkyCoord, HADec, EarthLocation
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from panoptes.data.observations import ObservationPathInfo
from photutils import segmentation
from photutils.utils import calc_total_error
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dateutil.tz import UTC
from dateutil.parser import parse as parse_date

import papermill as pm

from panoptes.pipeline.settings import PipelineParams
from panoptes.pipeline.utils import sources
from panoptes.pipeline.utils.gcp.bigquery import get_bq_clients

from panoptes.utils.images import fits as fits_utils
from panoptes.utils.images import bayer


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


def process_notebook(fits_path: str,
                     input_notebook: Path,
                     output_dir: Path = Path('.'),
                     settings: Settings = None
                     ) -> str:
    print(f'Starting image processing for {fits_path} in {output_dir!r}')
    print(f'Checking if got a fits file at {fits_path}')
    try:
        path_info = ObservationPathInfo(path=fits_path)
    except ValueError as e:
        raise RuntimeError(f'Need a FITS file, got {fits_path}')

    # Set proper names for the image settings.
    image_settings = settings if settings is not None else Settings(output_dir=output_dir, files=dict(
        reduced_filename=f'{path_info.image_id}.fits.fz',
        sources_filename=f'{path_info.image_id}.sources.parquet'
    ))

    # Run papermill process to execute the notebook.
    out_notebook = f'{output_dir}/{path_info.get_full_id()}-processing.ipynb'
    try:
        pm.execute_notebook(str(input_notebook),
                            str(out_notebook),
                            parameters=dict(
                                fits_path=str(fits_path),
                                output_dir=str(image_settings.output_dir),
                                image_settings=image_settings.model_dump_json()
                            ),
                            progress_bar=False,
                            )

    except Exception as e:
        print(f'Problem processing image for {fits_path}: {e!r}')

    return out_notebook


def save_fits(filename, data_list, header, force_new=False):
    hdul = fits.HDUList()
    for name, d in data_list.items():
        hdu = fits.ImageHDU(d, header=header)
        hdu.name = name.upper()
        hdul.append(hdu)

    hdul.writeto(filename, overwrite=force_new)
    print(f'Saved {len(data_list)} dataset(s) to {filename}')


def get_metadata(settings: Settings, path_info: ObservationPathInfo) -> dict:
    header = fits.getheader(settings.files.reduced_filename)

    # Puts metadata into better structures.
    metadata = extract_metadata(header, path_info)
    wcs_meta = WCS(header).to_header(relax=True)

    obstime = metadata['image']['time']

    # Clean up the coordinates and get the HA and AltAz.
    radec_coord = SkyCoord(ra=wcs_meta['CRVAL1'],
                           dec=wcs_meta['CRVAL2'],
                           unit='deg',
                           frame='icrs',
                           obstime=obstime,
                           location=EarthLocation(lon=header['LONG-OBS'],
                                                  lat=header['LAT-OBS'],
                                                  height=header['ELEV-OBS']))
    hadec_coord = radec_coord.transform_to(HADec)

    # Update metadata with coordinate info.
    metadata['image']['coordinates'] = {
        'ra': radec_coord.ra.value,
        'dec': radec_coord.dec.value,
        'ha': hadec_coord.ha.value,
        'ha_deg': hadec_coord.ha.to('deg').value,
        'alt': hadec_coord.altaz.alt.value,
        'az': hadec_coord.altaz.az.value,
        'airmass': hadec_coord.altaz.secz.value,
    }

    metadata['image']['image_type'] = 'SCIENCE'
    metadata['sequence']['image_type'] = 'SCIENCE'

    return metadata


def extract_metadata(header, path_info) -> dict:
    """Get the metadata from a FITS image."""
    try:
        measured_rggb = [float(x) for x in header.get('MEASRGGB', '0 0 0 0').split(' ')]
        file_date = path_info.image_time.to_datetime(timezone=UTC)
        camera_date = parse_date(header.get('DATE-OBS', path_info.image_time)).replace(tzinfo=UTC)

        unit_info = dict(
            unit_id=path_info.unit_id,
            latitude=header.get('LAT-OBS'),
            longitude=header.get('LONG-OBS'),
            elevation=float(header.get('ELEV-OBS')),
            name=header.get('OBSERVER')
        )

        sequence_info = dict(
            sequence_id=path_info.sequence_id,
            sequence_time=path_info.sequence_time.to_datetime(timezone=UTC),
            coordinates=dict(
                airmass=header.get('AIRMASS'),
                mount_dec=header.get('DEC-MNT'),
                mount_ra=header.get('RA-MNT'),
                mount_ha=header.get('HA-MNT'),
            ),
            camera=dict(
                camera_id=path_info.camera_id,
                lens_serial_number=header.get('INTSN'),
                serial_number=str(header.get('CAMSN')),
            ),
            imagew=int(header.get('IMAGEW', 0)),
            imageh=int(header.get('IMAGEH', 0)),
            field_name=header.get('FIELD', ''),
            software_version=header.get('CREATOR', ''),
        )

        image_info = dict(
            uid=path_info.get_full_id(sep='_'),
            camera=dict(
                blue_balance=float(header.get('BLUEBAL')),
                circconf=float(header.get('CIRCCONF', '0.').split(' ')[0]),
                colortemp=float(header.get('COLORTMP')),
                dateobs=camera_date,
                exptime=float(header.get('EXPTIME')),
                iso=header.get('ISO'),
                measured_b=measured_rggb[3],
                measured_ev1=float(header.get('MEASEV')),
                measured_ev2=float(header.get('MEASEV2')),
                measured_g1=measured_rggb[1],
                measured_g2=measured_rggb[2],
                measured_r=measured_rggb[0],
                red_balance=float(header.get('REDBAL')),
                temperature=float(header.get('CAMTEMP', 0).split(' ')[0]),
                white_lvln=header.get('WHTLVLN'),
                white_lvls=header.get('WHTLVLS'),
            ),
            environment=dict(
                moonfrac=float(header.get('MOONFRAC')),
                moonsep=float(header.get('MOONSEP')),
            ),
            file_creation_date=file_date,
            image_time=path_info.image_time.to_datetime(timezone=UTC),
        )

    except Exception as e:
        print(f'Error in extracting metadata: {e!r}')
        raise e

    print(f'Metadata extracted from header')
    return dict(unit=unit_info, sequence=sequence_info, image=image_info)


def match_sources(detected_sources: pandas.DataFrame, solved_wcs0: WCS, settings: Settings,
                  image_edge: int = 10) -> pandas.DataFrame:
    print(f'Matching {len(detected_sources)} sources to wcs.')
    catalog_filename = settings.params.catalog.catalog_filename
    if catalog_filename and catalog_filename.exists():
        print(f'Using catalog from {settings.params.catalog.catalog_filename}')
        catalog_sources = pd.read_parquet(settings.params.catalog.catalog_filename)
    else:
        print(f'Getting catalog sources from bigquery for WCS')
        # BQ client.
        bq_client, bqstorage_client = get_bq_clients()
        vmag_limits = settings.params.catalog.vmag_limits
        catalog_sources = sources.get_stars_from_wcs(solved_wcs0,
                                                     bq_client=bq_client,
                                                     bqstorage_client=bqstorage_client,
                                                     vmag_min=vmag_limits[0],
                                                     vmag_max=vmag_limits[1],
                                                     )
    print(f'Matching sources to catalog for {len(detected_sources)} sources')
    matched_sources = sources.get_catalog_match(detected_sources,
                                                wcs=solved_wcs0,
                                                catalog_stars=catalog_sources,
                                                ra_column='photutils_sky_centroid_ra',
                                                dec_column='photutils_sky_centroid_dec',
                                                max_separation_arcsec=settings.params.catalog.max_separation_arcsec
                                                )
    # Drop matches near border
    print(f'Filtering sources near within {image_edge} pixels of '
          f'{settings.params.camera.image_width}x{settings.params.camera.image_height}')
    matched_sources = matched_sources.query(
        f'catalog_wcs_x_int > {image_edge} and '
        f'catalog_wcs_x_int < {settings.params.camera.image_width - image_edge} and '
        f'catalog_wcs_y_int > {image_edge} and '
        f'catalog_wcs_y_int < {settings.params.camera.image_height - image_edge}'
    ).copy()
    print(f'Found {len(matched_sources)} matching sources')

    # There should not be too many duplicates at this point and they are returned in order
    # of catalog separation, so we take the first.
    duplicates = matched_sources.duplicated('picid', keep='first')
    print(f'Found {len(matched_sources[duplicates])} duplicate sources')

    # Mark which ones were duplicated.
    matched_sources.loc[:, 'catalog_match_duplicate'] = False
    dupes = matched_sources.picid.isin(matched_sources[duplicates].picid)
    matched_sources.loc[dupes, 'catalog_match_duplicate'] = True

    # Filter out duplicates.
    matched_sources = matched_sources.loc[~duplicates].copy()
    print(f'Found {len(matched_sources)} matching sources after removing duplicates')

    # Add some binned fields that we can use for table partitioning.
    matched_sources['catalog_dec_bin'] = matched_sources.catalog_dec.astype('int')
    matched_sources['catalog_ra_bin'] = matched_sources.catalog_ra.astype('int')

    # Precompute some columns.
    matched_sources['catalog_gaia_bg_excess'] = matched_sources.catalog_gaiabp - matched_sources.catalog_gaiamag
    matched_sources['catalog_gaia_br_excess'] = matched_sources.catalog_gaiabp - matched_sources.catalog_gaiarp
    matched_sources['catalog_gaia_rg_excess'] = matched_sources.catalog_gaiarp - matched_sources.catalog_gaiamag

    return matched_sources


def detect_sources(solved_wcs0, reduced_data, combined_bg_data, combined_bg_residual_data,
                   settings: Settings):
    print('Detecting sources in image')
    threshold = (settings.params.catalog.detection_threshold * combined_bg_residual_data)
    kernel = convolution.Gaussian2DKernel(3 * gaussian_fwhm_to_sigma)
    kernel.normalize()
    image_segments = segmentation.detect_sources(reduced_data,
                                                 threshold,
                                                 npixels=settings.params.catalog.num_detect_pixels,
                                                 mask=reduced_data.mask
                                                 )
    print(f'De-blending image segments')
    deblended_segments = segmentation.deblend_sources(reduced_data,
                                                      image_segments,
                                                      npixels=settings.params.catalog.num_detect_pixels,
                                                      nlevels=32,
                                                      contrast=0.01)
    print(
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
    print('Building source catalog for deblended_segments')
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


def plate_solve(settings: Settings, filename=None, timeout=30, **kwargs):
    filename = filename or settings.files.reduced_filename
    print(f'Plate solving {filename}')

    # Add custom options for solving.
    options = [
        '--fits-image',
        '--scale-low', '10',
        '--scale-high', '20',
        '--scale-units', 'degw',
        '--radius', '15',
        '--guess-scale',
        '--no-background-subtraction',
        '--cpulimit', str(timeout),
        '--no-verify',
        '--crpix-center',
        '--temp-axy',
        '--index-xyls', 'none',
        '--solved', 'none',
        '--match', 'none',
        '--rdls', 'none',
        '--corr', filename.with_suffix('.corr'),
        '--downsample', '4',
        '--tweak-order', '4',
        '--no-plots',
    ]

    solved_headers = fits_utils.get_solve_field(str(filename),
                                                skip_solved=False,
                                                solve_opts=options,
                                                timeout=300, **kwargs)
    solved_path = solved_headers.pop('solved_fits_file')
    print(f'Solving completed successfully for {solved_path}')
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
