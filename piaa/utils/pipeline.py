import os
import re
import shutil
import subprocess

from contextlib import suppress

import h5py
import numpy as np
import pandas as pd

from scipy import linalg
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time
from astropy.stats import sigma_clipped_stats, sigma_clip
from scipy.signal import savgol_filter

from tqdm import tqdm, tqdm_notebook

from dateutil.parser import parse as date_parse

from photutils import DAOStarFinder

from pocs.utils.images import fits as fits_utils
from piaa.utils import helpers
from piaa.utils import plot
from piaa.utils.postgres import get_cursor

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def normalize(cube):
    return (cube.T / cube.sum(1)).T


def lookup_sources_for_observation(fits_files=None,
                                   filename=None,
                                   force_new=False,
                                   cursor=None,
                                   use_intersection=False,
                                   **kwargs
                                   ):

    if force_new:
        logger.info(f'Forcing a new source file')
        with suppress(FileNotFoundError):
            os.remove(filename)

    try:
        logger.info(f'Using existing source file: {filename}')
        observation_sources = pd.read_csv(filename, parse_dates=True)
        observation_sources['obstime'] = pd.to_datetime(observation_sources.obstime)
        observation_sources.rename(columns={'id': 'picid'}, inplace=True)

    except FileNotFoundError:
        if not cursor:
            cursor = get_cursor(port=5433, db_name='v6', db_user='postgres')

        logger.info(f'Looking up sources in {len(fits_files)} files')
        observation_sources = None

        # Lookup the point sources for all frames
        for fn in tqdm(fits_files):
            point_sources = lookup_point_sources(
                fn,
                force_new=force_new,
                cursor=cursor,
                **kwargs
            )
            header = fits_utils.getheader(fn)
            obstime = Time(pd.to_datetime(os.path.basename(fn).split('.')[0]))
            exptime = header['EXPTIME'] * u.second

            obstime += (exptime / 2)

            point_sources['obstime'] = obstime.datetime
            point_sources['exptime'] = exptime
            point_sources['airmass'] = header['AIRMASS']
            point_sources['file'] = os.path.basename(fn)
            point_sources['picid'] = point_sources.index

            if observation_sources is not None:
                if use_intersection:

                    idx_intersection = observation_sources.index.intersection(point_sources.index)
                    logger.info(f'Num sources in intersection: {len(idx_intersection)}')
                    observation_sources = pd.concat([observation_sources.loc[idx_intersection],
                                                     point_sources.loc[idx_intersection]],
                                                    join='inner')
                else:
                    observation_sources = pd.concat([observation_sources, point_sources])
            else:
                observation_sources = point_sources

        observation_sources.to_csv(filename)

    observation_sources.set_index(['obstime'], inplace=True)
    return observation_sources


def lookup_point_sources(fits_file,
                         catalog_match=True,
                         method='sextractor',
                         force_new=False,
                         **kwargs
                         ):
    """ Extract point sources from image

    Args:
        fits_file (str, optional): Path to FITS file to search for stars.
        force_new (bool, optional): Force a new catalog to be created,
            defaults to False

    Raises:
        error.InvalidSystemCommand: Description
    """
    if catalog_match or method == 'tess_catalog':
        fits_header = fits_utils.getheader(fits_file)
        wcs = WCS(fits_header)
        assert wcs is not None and wcs.is_celestial, logger.warning("Need a valid WCS")

    logger.info("Looking up sources for {}".format(fits_file))

    lookup_function = {
        'sextractor': _lookup_via_sextractor,
        'tess_catalog': _lookup_via_tess_catalog,
        'photutils': _lookup_via_photutils,
    }

    # Lookup our appropriate method and call it with the fits file and kwargs
    try:
        logger.info("Using {} method {}".format(method, lookup_function[method]))
        point_sources = lookup_function[method](fits_file, force_new=force_new, **kwargs)
    except Exception as e:
        logger.error("Problem looking up sources: {}".format(e))
        raise Exception("Problem lookup up sources: {}".format(e))

    if catalog_match:
        point_sources = get_catalog_match(point_sources, wcs, **kwargs)

    # Change the index to the picid
    point_sources.set_index('id', inplace=True)
    point_sources.index.rename('picid', inplace=True)

    # Remove those with more than one entry
    counts = point_sources.x.groupby('picid').count()
    single_entry = counts == 1
    single_index = single_entry.loc[single_entry].index
    unique_sources = point_sources.loc[single_index]

    return unique_sources


def get_catalog_match(point_sources, wcs, table='full_catalog', **kwargs):
    assert point_sources is not None

    # Get coords from detected point sources
    stars_coords = SkyCoord(
        ra=point_sources['ra'].values * u.deg,
        dec=point_sources['dec'].values * u.deg
    )

    # Lookup stars in catalog
    catalog_stars = helpers.get_stars_from_footprint(
        wcs.calc_footprint(),
        cursor_only=False,
        table=table,
        **kwargs
    )

    # Get coords for catalog stars
    catalog_coords = SkyCoord(
        ra=catalog_stars['ra'] * u.deg,
        dec=catalog_stars['dec'] * u.deg
    )

    # Do catalog matching
    idx, d2d, d3d = match_coordinates_sky(stars_coords, catalog_coords)

    # Get some properties from the catalog
    point_sources['id'] = catalog_stars[idx]['id']
    # point_sources['twomass'] = catalog_stars[idx]['twomass']
    point_sources['tmag'] = catalog_stars[idx]['tmag']
    point_sources['vmag'] = catalog_stars[idx]['vmag']
    point_sources['catalog_sep_arcsec'] = d2d.to(u.arcsec).value

    return point_sources


def _lookup_via_sextractor(fits_file, sextractor_params=None, *args, **kwargs):
    # Write the sextractor catalog to a file
    base_dir = os.path.dirname(fits_file)
    source_dir = os.path.join(base_dir, 'sextractor')
    os.makedirs(source_dir, exist_ok=True)

    img_id = os.path.splitext(os.path.basename(fits_file))[0]

    source_file = os.path.join(source_dir, f'point_sources_{img_id}.cat')

    # sextractor can't handle compressed data
    if fits_file.endswith('.fz'):
        fits_file = fits_utils.funpack(fits_file)

    logger.info("Point source catalog: {}".format(source_file))

    if not os.path.exists(source_file) or kwargs.get('force_new', False):
        logger.info("No catalog found, building from sextractor")
        # Build catalog of point sources
        sextractor = shutil.which('sextractor')
        if sextractor is None:
            sextractor = shutil.which('sex')
            if sextractor is None:
                raise Exception('sextractor not found')

        if sextractor_params is None:
            sextractor_params = [
                '-c', '{}/PIAA/resources/conf_files/sextractor/panoptes.sex'.format(
                    os.getenv('PANDIR')),
                '-CATALOG_NAME', source_file,
            ]

        logger.info("Running sextractor...")
        cmd = [sextractor, *sextractor_params, fits_file]
        logger.info(cmd)

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, timeout=60, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception("Problem running sextractor: {}".format(e))

    # Read catalog
    point_sources = Table.read(source_file, format='ascii.sextractor')

    # Remove the point sources that sextractor has flagged
    # if 'FLAGS' in point_sources.keys():
    #    point_sources = point_sources[point_sources['FLAGS'] == 0]
    #    point_sources.remove_columns(['FLAGS'])

    # Rename columns
    point_sources.rename_column('XPEAK_IMAGE', 'x')
    point_sources.rename_column('YPEAK_IMAGE', 'y')

    # Filter point sources near edge
    # w, h = data[0].shape
    w, h = (3476, 5208)

    stamp_size = 60

    top = point_sources['y'] > stamp_size
    bottom = point_sources['y'] < w - stamp_size
    left = point_sources['x'] > stamp_size
    right = point_sources['x'] < h - stamp_size

    point_sources = point_sources[top & bottom & right & left].to_pandas()
    point_sources.columns = [
        'ra', 'dec',
        'x', 'y',
        'x_image', 'y_image',
        'background',
        'flux_best', 'fluxerr_best',
        'mag_best', 'magerr_best',
        'flux_aper', 'fluxerr_aper',
        'mag_aper', 'magerr_aper',
        'flux_max',
        'fwhm_image',
        'flags',
    ]

    return point_sources


def _lookup_via_tess_catalog(fits_file, wcs=None, *args, **kwargs):
    wcs_footprint = wcs.calc_footprint()
    logger.info("WCS footprint: {}".format(wcs_footprint))

    # Get stars from TESS catalog
    point_sources = helpers.get_stars_from_footprint(
        wcs_footprint,
        cursor_only=False,
        table=kwargs.get('table', 'full_catalog')
    )

    # Get x,y coordinates
    star_pixels = wcs.all_world2pix(point_sources['ra'], point_sources['dec'], 0)
    point_sources['x'] = star_pixels[0]
    point_sources['y'] = star_pixels[1]

    point_sources.add_index(['id'])
    point_sources = point_sources.to_pandas()

    return point_sources


def _lookup_via_photutils(fits_file, wcs=None, *args, **kwargs):
    data = fits.getdata(fits_file) - 2048  # Camera bias
    mean, median, std = sigma_clipped_stats(data)

    fwhm = kwargs.get('fwhm', 3.0)
    threshold = kwargs.get('threshold', 3.0)

    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources = daofind(data - median).to_pandas()

    sources.rename(columns={
        'xcentroid': 'x',
        'ycentroid': 'y',
    }, inplace=True)

    if wcs is None:
        header = fits_utils.getheader(fits_file)
        wcs = WCS(header)

    coords = wcs.all_pix2world(sources['x'], sources['y'], 1)

    sources['ra'] = coords[0]
    sources['dec'] = coords[1]

    return sources


def create_stamp_slices(
    save_dir,
    fits_files,
    point_sources,
    stamp_size=(14, 14),
    force_new=False,
    verbose=False,
    *args, **kwargs
):
    """Create PANOPTES Stamp Cubes (PSC) for each point source.

    Creates a slice through the cube corresponding to a stamp and stores the
    subtracted data in the hdf5 table with key `stamp/<picid>`.

    Args:
        *args (TYPE): Description
        **kwargs (dict): `ipython_widget=True` can be passed to display progress
            within a notebook

    """

    errors = dict()

    num_frames = len(fits_files)
    unit_id, cam_id, seq_time = fits_utils.getval(fits_files[0], 'SEQID').split('_')
    unit_id = re.match(r'.*(PAN\d\d\d).*', unit_id)[1]
    sequence = '_'.join([unit_id, cam_id, seq_time])

    logger.info("{} files found for {}".format(num_frames, sequence))

    stamps_fn = os.path.join(
        save_dir,
        sequence.replace('/', '_') + '.hdf5'
    )
    logger.info("Creating stamps file: {}".format(stamps_fn))

    if force_new is False and os.path.exists(stamps_fn):
        logger.info("Looking for existing stamps file")
        try:
            assert os.path.exists(stamps_fn)
            stamps = h5py.File(stamps_fn)
            logger.info("Returning existing stamps file")
            return stamps_fn
        except FileNotFoundError:
            pass
    else:
        # Make sure to delete existing
        with suppress(FileNotFoundError):
            os.remove(stamps_fn)

    stamps = h5py.File(stamps_fn, 'a')

    # Currently a bug with DATE-OBS so use time from filename.
    image_times = list()
    for fn in fits_files:
        if 'pointing' in fn:
            continue

        try:
            fn_imagetime = Time(date_parse(os.path.basename(fn).split('.')[0])).mjd
            image_times.append(fn_imagetime)
        except Exception as e:
            logger.warning('Problem getting image time: {}'.format(e))

    # image_times = np.array(
    #    [Time(date_parse(fits.getval(fn, 'DATE-OBS'))).mjd for fn in fits_files])

    airmass = np.array([fits_utils.getval(fn, 'AIRMASS') for fn in fits_files])

    stamps.attrs['image_times'] = image_times
    stamps.attrs['airmass'] = airmass

    file_iterator = enumerate(fits_files)

    if verbose:
        if kwargs.get('notebook', False):
            file_iterator = tqdm_notebook(file_iterator, total=num_frames, desc='Looping files')
        else:
            file_iterator = tqdm(file_iterator, total=num_frames, desc='Looping files')

    for frame_idx, fn in file_iterator:
        # Get stamp data.
        with fits.open(fn) as hdu:
            hdu_idx = 0
            if fn.endswith('.fz'):
                logger.info("Using compressed FITS")
                hdu_idx = 1

            wcs = WCS(hdu[hdu_idx].header)
            d0 = hdu[hdu_idx].data

        star_iterator = point_sources.itertuples()
        if verbose:
            if kwargs.get('notebook', False):
                star_iterator = tqdm_notebook(star_iterator, total=len(point_sources),
                                              leave=False, desc="Point sources")
            else:
                star_iterator = tqdm(star_iterator, total=len(point_sources),
                                     leave=False, desc="Point sources")

        for star_row in star_iterator:
            star_id = str(star_row.Index)

            try:
                existing_sum = np.array(stamps[star_id]['data'][frame_idx]).sum()
                if star_id in stamps and existing_sum:
                    logger.info("Skipping {}, {} for having data: {}".format(star_id,
                                                                             frame_idx,
                                                                             existing_sum))
                    continue
            except KeyError:
                pass

            star_pos = wcs.all_world2pix(star_row.ra, star_row.dec, 1)

            # Get stamp data. If problem, mark for skipping in future.
            try:
                # This handles the RGGB pattern
                slice0 = helpers.get_stamp_slice(star_pos[0], star_pos[1], stamp_size=stamp_size)
                logger.debug("Slice for {} {}: {}".format(star_pos[0], star_pos[1], slice0))
                if not slice0:
                    logger.warning(
                        "Invalid slice for star_id {} on frame {}".format(star_id, frame_idx))
                    continue
                d1 = d0[slice0].flatten()

                if len(d1) == 0:
                    logger.warning('Bad slice for {}, skipping'.format(star_id))
                    continue
            except Exception as e:
                logger.warning("Problem with slice: {}".format(e))

            # Create group for stamp and add metadata
            try:
                psc_group = stamps[star_id]
            except KeyError:
                logger.debug("Creating new group for star {}".format(star_id))
                psc_group = stamps.create_group(star_id)
                # Stamp metadata
                try:
                    for col in point_sources.columns:
                        psc_group.attrs[col] = str(getattr(star_row, col))
                except Exception as e:
                    if str(e) not in errors:
                        logger.warning(e)
                        errors[str(e)] = True

            # Set the data for the stamp. Create PSC dataset if needed.
            try:
                # Assign stamp values
                psc_group['data'][frame_idx] = d1
            except KeyError:
                logger.debug("Creating new PSC dataset for {}".format(star_id))
                psc_size = (num_frames, len(d1))

                # Create the dataset
                stamp_dset = psc_group.create_dataset('data', psc_size, dtype='u2', chunks=True)

                # Assign the data
                stamp_dset[frame_idx] = d1
            except TypeError as e:
                # Sets the metadata. Create metadata dataset if needed.
                key = str(e) + star_id
                if key not in errors:
                    logger.info(e)
                    errors[key] = True

            try:
                psc_group['original_position'][frame_idx] = (star_row.x, star_row.y)
            except KeyError:
                logger.debug("Creating new metadata dataset for {}".format(star_id))
                metadata_size = (num_frames, 2)

                # Create the dataset
                metadata_dset = psc_group.create_dataset(
                    'original_position', metadata_size, dtype='u2', chunks=True)

                # Assign the data
                metadata_dset[frame_idx] = (star_row.x, star_row.y)

            stamps.flush()

        if errors:
            logger.warning(errors)

    return stamps_fn


def get_psc(picid, stamps, frame_slice=None):
    try:
        psc = np.array(stamps[picid]['data'])
    except KeyError:
        raise Exception("{} not found in the stamp collection.".format(picid))

    if frame_slice is not None:
        psc = psc[frame_slice]

    return psc


def find_similar_stars(
        picid,
        stamps,
        csv_file=None,
        camera_bias=2048,
        num_refs=100,
        snr_limit=10,
        show_progress=True,
        force_new=False,
        *args, **kwargs):
    """ Get all variances for given target

    Args:
        stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
        i(int): Index of target PIC
    """
    logger.info("Finding similar stars for PICID {}".format(picid))

    if force_new and csv_file and os.path.exist(csv_file):
        logger.info("Forcing new file for {}".format(picid))
        with suppress(FileNotFoundError):
            os.remove(csv_file)

    try:
        df0 = pd.read_csv(csv_file, index_col=[0])
        logger.info("Found existing csv file: {}".format(df0))
        return df0
    except Exception:
        pass

    data = dict()

    logger.info("Getting Target PSC and subtracting bias")
    psc0 = get_psc(picid, stamps, **kwargs) - camera_bias
    logger.info("Target PSC shape: {}".format(psc0.shape))
    num_frames = psc0.shape[0]

    # Normalize
    logger.info("Normalizing target for {} frames".format(num_frames))
    normalized_psc0 = np.zeros_like(psc0, dtype='f4')

    good_frames = []
    for frame_index in range(num_frames):
        try:
            if psc0[frame_index].sum() > 0.:
                # Normalize and store frame
                normalized_psc0[frame_index] = psc0[frame_index] / psc0[frame_index].sum()

                # Save frame index
                good_frames.append(frame_index)
            else:
                logger.warning("Sum for target frame {} is 0".format(frame_index))
        except RuntimeWarning:
            logger.warning("Skipping frame {}".format(frame_index))

    iterator = enumerate(list(stamps.keys()))
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(stamps),
            desc="Finding similar",
            leave=False
        )

    for i, source_index in iterator:
        # Skip low SNR (if we know SNR)
        try:
            snr = float(stamps[source_index].attrs['snr'])
            if snr < snr_limit:
                logger.info("Skipping PICID {}, low snr {:.02f}".format(source_index, snr))
                continue
        except KeyError as e:
            logger.debug("No source in table: {}".format(picid))
            pass

        try:
            psc1 = get_psc(source_index, stamps, **kwargs) - camera_bias
        except Exception:
            continue

        normalized_psc1 = np.zeros_like(psc1, dtype='f4')

        # Normalize
        for frame_index in good_frames:
            if psc1[frame_index].sum() > 0.:
                normalized_psc1[frame_index] = psc1[frame_index] / psc1[frame_index].sum()

        # Store in the grid
        try:
            v = ((normalized_psc0 - normalized_psc1) ** 2).sum()
            data[source_index] = v
        except ValueError as e:
            logger.info("Skipping invalid stamp for source {}: {}".format(source_index, e))

    df0 = pd.DataFrame(
        {'v': list(data.values())},
        index=list(data.keys())).sort_values(by='v')

    if csv_file:
        df0[:num_refs].to_csv(csv_file)

    return df0


def get_ideal_full_coeffs(stamp_collection):

    num_frames = stamp_collection.shape[1]
    num_pixels = stamp_collection.shape[2]

    target_frames = stamp_collection[0].flatten()
    refs_frames = stamp_collection[1:].reshape(-1, num_frames * num_pixels).T

    coeffs = linalg.lstsq(refs_frames, target_frames)

    return coeffs


def get_ideal_full_psc(stamp_collection, coeffs):
    refs = stamp_collection[1:]
    created_frame = (refs.T * coeffs).sum(2).T
    return created_frame


def get_aperture_sums(psc0,
                      psc1,
                      image_times,
                      use_bright_pixels=False,
                      sigma_mask_aperture=False,
                      sigma_mask_threshold=2.5,
                      adaptive_aperture=False,
                      num_stds=2,
                      aperture_size=None,
                      readout_noise=10.5,
                      gain=1.5,
                      separate_green=False,
                      subtract_back=False,
                      plot_apertures=False,
                      aperture_plot_path=None,
                      picid=None,
                      ):
    """Perform differential aperture photometry on the given PSCs.

    `psc0` and `psc1` are Postage Stamp Cubes (PSC) of N frames x M
    pixels, where M = width x height of the stamp and is assumed to be
    square.

    For each N frame, an aperture is placed around the source in `psc0`
    and the corresponding pixel location in `psc1`. This aperture cutout
    is then split on color channels and for each channel the sum of
    the target, the sum of the reference, and the difference is given.

    ..todo::

        Adaptive Aperture:

    Args:
        psc0 (`numpy.array`): An NxM cube of source postage stamps.
        psc1 (`numpy.array`): An NxM cube to be used as the comparison.
        image_times (list(`datetime`)): A list of `datetime.datetime` objects to
            be used for an index.
        adaptive_aperture (bool): An adaptive aperture is used, in which the brightest
            pixels from each channel are used. See note for details.
        num_stds (int): Number of standard deviations to use for adaptive aperture,
            default 2.
        aperture_size (int): An aperture around the source that is used
            for photometry, default None uses entire stamp if `adaptive_aperture=True`.
        readout_noise (float): Readout noise in e- / pixel, default 10.5.
        gain (float): Camera gain in e- / ADU, default 1.5. TODO: Alter this for per
            channel gain.
        separate_green (bool): If separate green color channels should be created,
            default False. If True, the G2 pixel is marked as `c`.
        subtract_back (bool, optional): If a background annulus should be removed
            from the sum, default False.
        plot_apertures (bool, optional): If a figure should be generated showing
            each of the aperture stamps, default False.

    Returns:
        `pandas.DataFrame`: A dataframe with `color`, `target`, and `reference`.
            columns.

    """
    num_frames, stamp_size = psc0.shape

    stamp_side = int(np.sqrt(stamp_size))

    if not adaptive_aperture:
        try:
            single_frame = psc0[0].reshape(stamp_side, stamp_side)

            rgb_stamp_masks = helpers.get_rgb_masks(
                single_frame,
                force_new=True,
                separate_green=separate_green
            )
            logger.debug('RGB stamp_masks created')
        except ValueError as e:
            logger.warning(f"Can't make stamp masks {e!r}")

    apertures = list()
    diff = list()
    for frame_idx, image_time in zip(range(num_frames), image_times):

        # Get target and reference stamp for this frame
        t0 = psc0[frame_idx].reshape(stamp_side, stamp_side)
        i0 = psc1[frame_idx].reshape(stamp_side, stamp_side)

        if use_bright_pixels:
            logger.debug(f'Using bright pixels')
            
            # Get the bright pixels but don't sum yet
            t_bright_pixels = helpers.get_bright_pixels(t0, sum=False)
            i_bright_pixels = helpers.get_bright_pixels(i0, sum=False)
            
            for color in 'rgb':
                
                # Get the sum in electrons
                t_sum = t_bright_pixels[color].sum()
                i_sum = i_bright_pixels[color].sum()

                # Add the noise
                target_photon_noise = np.sqrt(t_sum)
                ideal_photon_noise = np.sqrt(i_sum)
                
                pixel_count = len(t_bright_pixels[color])

                readout = readout_noise * pixel_count

                target_total_noise = np.sqrt(target_photon_noise**2 + readout**2)
                ideal_total_noise = np.sqrt(ideal_photon_noise**2 + readout**2)

                # Record the values.
                diff.append({
                    'color': color,
                    'target': t_sum,
                    'target_err': target_total_noise,
                    'reference': i_sum,
                    'reference_err': ideal_total_noise,
                    'obstime': image_time,
                })
        elif sigma_mask_aperture:
            logger.debug(f'Using sigma mask aperture with σ={sigma_mask_threshold}')
            
            i0_rgb_data = helpers.get_rgb_data(i0)
            
            # We use the reference to make the aperture
            masked_stamps = helpers.make_sigma_masked_stamps(i0_rgb_data, sigma_thresh=sigma_mask_threshold) 
            for color in 'rgb':
                # Make the mask with sigma aperture
                t0_aperture = np.ma.array(t0, mask=masked_stamps[color].mask)
                i0_aperture = np.ma.array(i0, mask=masked_stamps[color].mask)
                
                # Get the sum in electrons
                t_sum = (t0_aperture * gain).sum()
                i_sum = (i0_aperture * gain).sum()

                # Add the noise
                target_photon_noise = np.sqrt(t_sum)
                ideal_photon_noise = np.sqrt(i_sum)

                readout = readout_noise * i0_aperture.count()

                target_total_noise = np.sqrt(target_photon_noise**2 + readout**2)
                ideal_total_noise = np.sqrt(ideal_photon_noise**2 + readout**2)

                # Record the values.
                diff.append({
                    'color': color,
                    'target': t_sum,
                    'target_err': target_total_noise,
                    'reference': i_sum,
                    'reference_err': ideal_total_noise,
                    'obstime': image_time,
                })
            
        elif adaptive_aperture:
            logger.debug(f'Using adaptive apertures')
            pixel_locations = helpers.get_adaptive_aperture_pixels(target_stamp=i0,
                                                                   frame_idx=frame_idx,
                                                                   make_plots=plot_apertures,
                                                                   target_dir=aperture_plot_path,
                                                                   picid=picid,
                                                                   num_stds=num_stds
                                                                   )
            logger.debug(f'Frame {frame_idx} pixel locations: {pixel_locations}')

            for color, pixel_loc in pixel_locations.items():
                logger.debug(color, pixel_loc)
                target_pixel_values = np.array([t0[loc[0], loc[1]] for loc in pixel_loc])
                ideal_pixel_values = np.array([i0[loc[0], loc[1]] for loc in pixel_loc])

                # Get the sum in electrons
                t_sum = (target_pixel_values * gain).sum()
                i_sum = (ideal_pixel_values * gain).sum()

                # Add the noise
                target_photon_noise = np.sqrt(t_sum)
                ideal_photon_noise = np.sqrt(i_sum)

                readout = readout_noise * len(pixel_loc)

                target_total_noise = np.sqrt(target_photon_noise**2 + readout**2)
                ideal_total_noise = np.sqrt(ideal_photon_noise**2 + readout**2)

                # Record the values.
                diff.append({
                    'color': color,
                    'target': t_sum,
                    'target_err': target_total_noise,
                    'reference': i_sum,
                    'reference_err': ideal_total_noise,
                    'obstime': image_time,
                })

        elif aperture_size:
            # NOTE: Bad "centroiding" here
            y_pos, x_pos = np.argwhere(t0 == t0.max())[0]
            aperture_position = (x_pos, y_pos)
            center_color = helpers.pixel_color(x_pos, y_pos)

            slice0 = None
            if aperture_size != stamp_side:
                logger.debug('Aperture Position: {} Size: {} Frame: {} Color: {}'.format(
                    aperture_position, aperture_size, frame_idx, center_color))
                slice0 = helpers.get_stamp_slice(x_pos, y_pos, stamp_size=(
                    aperture_size, aperture_size), ignore_superpixel=True)
                logger.debug(f'Slice for aperture: {slice0}')

            for color, mask in rgb_stamp_masks.items():

                # Get color mask data from target and reference
                t1 = np.ma.array(t0, mask=~mask)
                i1 = np.ma.array(i0, mask=~mask)

                # Make apertures
                if slice0:
                    try:
                        t3 = t1[slice0]
                        i3 = i1[slice0]
                    except Exception as e:
                        logger.debug(e)
                        continue

                    if plot_apertures:
                        apertures.append((t3, image_time, center_color))
                    logger.debug(t3)
                else:
                    t3 = t1
                    i3 = i1

                if subtract_back:
                    mean, median, std = sigma_clipped_stats(t3)
                    logger.info("Average sky background for {}: {:5.2f}: {:5.2f}".format(
                        color, mean, t3.sum()))

                    t3 = t3 - mean
                    logger.info(t3.sum())

                t_sum = t3.sum()
                i_sum = i3.sum()

                # Record the values.
                diff.append({
                    'color': color,
                    'target': t_sum,
                    'reference': i_sum,
                    'obstime': image_time,
                })
        else:
            raise UserWarning(f'adaptive_aperture is false but no aperture_size given')

    # Light-curve dataframe
    lc0 = pd.DataFrame(diff).set_index(['obstime'])

    if plot_apertures:
        os.makedirs(aperture_plot_path, exist_ok=True)
        plot.make_apertures_plot(apertures, output_dir=aperture_plot_path)

    return lc0


def get_imag(x, t=1):
    """Instrumental magnitude.

    Args:
        x (float|list(float)): Flux values.
        t (int, optional): Exposure time.

    Returns:
        float|list(float): Instrumental magnitudes.
    """
    return -2.5 * np.log10(x / t)


def normalize_lightcurve(lc0, method='median', use_frames=None):
    """Normalize the lightcurve data, including errors.

    Args:
        lc0 (`pandas.DataFrame`): Dataframe with light curve values.
        method (str, optional): The normalization method used, either `median` (default) or `mean`.
        use_frames (None, optional): A `slice` object to select frame for normalization, e.g. the
            pre-ingress or post-egress frames.

    Returns:
        `pandas.DataFrame`: A copy of the dataframe with normalized light curve values and error.
    """
    # Make a copy
    lc1 = lc0.copy()

    methods = {
        'median': lambda x: np.median(x),
        'mean': lambda x: np.mean(x)
    }

    use_method = methods[method]
    if use_frames is None:
        use_frames = slice(None)

    data_to_normalize = lc1[use_frames].groupby('color')

    for field in ['reference', 'target']:
        field_to_normalize = data_to_normalize[field]

        # Apply the normalization.
        normalization_values = field_to_normalize.apply(use_method)

        for color, norm_value in normalization_values.iteritems():
            logging.info(f"{field} {color} μ={norm_value:.04f}")

            # Get the raw values.
            raw_values = lc1.loc[lc1.color == color, (f'{field}')]
            raw_error = lc1.loc[lc1.color == color, (f'{field}_err')]

            # Replace with normalized versions.
            lc1.loc[lc1.color == color, (f'{field}')] = (raw_values / norm_value)
            lc1.loc[lc1.color == color, (f'{field}_err')] = (raw_error / norm_value)

    return lc1

def get_diff_flux(lc0, 
                  smooth=False,
                  savgol_polyorder=None,
                  savgol_sigma=3,
                  sigma_cutoff=None
                 ):
    
    lc1 = lc0.copy()
    lc1['flux'] = np.nan
    lc1['flux_err'] = np.nan
    
    color_dfs = list()
    for i, color in enumerate('rgb'):
        # Get the normalized flux for each channel
        color_data = lc1.loc[lc1.color == color]

        # Target and error
        t0 = color_data.target
        t0_err = color_data.target_err

        # Reference and error
        r0 = color_data.reference
        r0_err = color_data.reference_err

        # Get the differential flux and error
        flux = t0 / r0
        flux_err = np.sqrt((t0_err / t0)**2 + (r0_err / r0)**2)
        flux_index = color_data.index
        
        if savgol_polyorder:
            if savgol_polyorder % 2 == 0:
                window_size = savgol_polyorder+1
            else:
                window_size = savgol_polyorder+2
                
            filter0 = savgol_filter(flux, window_size, polyorder=savgol_polyorder)
            flux = (flux - filter0)
            
            # Clip the filtered
            flux = sigma_clip(flux, sigma=savgol_sigma)
            
            # Add back the filter
            flux = flux + filter0
            
            flux_err = np.ma.array(flux_err, mask=flux.mask)
            flux_index = np.ma.array(color_data.index, mask=flux.mask)

        if sigma_cutoff:
            # Sigma clip the differential flux
            flux = sigma_clip(flux, sigma=sigma_cutoff)
            flux_err = np.ma.array(flux_err, mask=flux.mask)
            flux_index = np.ma.array(color_data.index, mask=flux.mask, dtype=bool)
            
        # Basic correction
        if smooth:
            window_size = len(flux)
            if window_size % 2 == 0:
                window_size -= 1
            smooth1 = savgol_filter(flux, window_size, polyorder=1)
            flux = (flux - smooth1) + 1
            
        lc1.loc[lc1.color == color, ('flux')] = flux.filled(np.nan)
        lc1.loc[lc1.color == color, ('flux_err')] = flux_err.filled(np.nan)
            
    return lc1, flux.mask
