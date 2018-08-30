import os
import shutil
import subprocess

from contextlib import suppress
from warnings import warn

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
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D, PartialOverlapError, NoOverlapError

from tqdm import tqdm, tqdm_notebook

from dateutil.parser import parse as date_parse

from photutils import aperture
from photutils import DAOStarFinder

from piaa.utils import helpers
from piaa.utils import plot

import logging
logger = logging.getLogger(__name__)


def normalize(cube):
    return (cube.T / cube.sum(1)).T


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
        wcs = WCS(fits_file)
        assert wcs is not None and wcs.is_celestial, logger.warning("Need a valid WCS")

    lookup_function = {
        'sextractor': _lookup_via_sextractor,
        'tess_catalog': _lookup_via_tess_catalog,
        'photutils': _lookup_via_photutils,
    }

    # Lookup our appropriate method and call it with the fits file and kwargs
    try:
        logger.info("Using {} method {}".format(method, lookup_function[method]))
        point_sources = lookup_function[method](fits_file, **kwargs)
    except Exception as e:
        logger.error("Problem looking up sources: {}".format(e))
        raise Exception("Problem lookup up sources: {}".format(e))

    if catalog_match:
        point_sources = get_catalog_match(point_sources, wcs, **kwargs)
        
    # Change the index to the picid
    point_sources.set_index('id', inplace=True)
        
    # Remove those with more than one entry
    counts = point_sources.x.groupby('id').count()
    single_entry = counts == 1
    single_index = single_entry.loc[single_entry].index
    unique_sources = point_sources.loc[single_entry.loc[single_entry].index]
    
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
    point_sources['twomass'] = catalog_stars[idx]['twomass']
    point_sources['tmag'] = catalog_stars[idx]['tmag']
    point_sources['vmag'] = catalog_stars[idx]['vmag']
    point_sources['d2d'] = d2d

    return point_sources


def _lookup_via_sextractor(fits_file, sextractor_params=None, *args, **kwargs):
    # Write the sextractor catalog to a file
    source_file = os.path.join(
        os.environ['PANDIR'],
        'psc',
        'point_sources_{}.cat'.format(fits_file.replace('/', '_'))
    )
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
    point_sources.rename_column('X_IMAGE', 'x')
    point_sources.rename_column('Y_IMAGE', 'y')

    # Add the SNR
    point_sources['SNR'] = point_sources['FLUX_AUTO'] / point_sources['FLUXERR_AUTO']

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
        'x', 'y',
        'ra', 'dec',
        'background',
        'flux_auto', 'flux_max', 'fluxerr_auto',
        'fwhm', 'flags', 'snr'
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
    data = fits.getdata(fits_file)
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
        wcs = WCS(fits_file)

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
    sequence = fits.getval(fits_files[0], 'SEQID')
    
    logging.info("{} files found for {}".format(num_frames, sequence))

    stamps_fn = os.path.join(
        save_dir,
        sequence.replace('/', '_') + '.hdf5'
    )
    logger.info("Creating stamps file: {}".format(stamps_fn))
    
    if force_new is False:
        logging.info("Looking for existing stamps file")
        try:
            assert os.path.exists(stamps_fn)
            stamps = h5py.File(stamps_fn)
            logging.info("Returning existing stamps file")
            return stamps_fn
        except FileNotFoundError:
            pass

    stamps = h5py.File(stamps_fn, 'a')

    image_times = np.array(
        [Time(date_parse(fits.getval(fn, 'DATE-OBS'))).mjd for fn in fits_files])
    airmass = np.array([fits.getval(fn, 'AIRMASS') for fn in fits_files])

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

            if star_id in stamps and np.array(stamps[star_id]['data'][frame_idx]).sum() > 1:
                logger.info("Skipping {} in frame {} for having data: {}".format(star_id, frame_idx, 
                                                                                  np.array(stamps[star_id]['data'][frame_idx]).sum()))
                continue

            star_pos = wcs.all_world2pix(star_row.ra, star_row.dec, 0)

            # Get stamp data. If problem, mark for skipping in future.
            try:
                # This handles the RGGB pattern
                slice0 = helpers.get_stamp_slice(star_pos[0], star_pos[1], stamp_size=stamp_size)
                d1 = d0[slice0].flatten()

                if len(d1) == 0:
                    logger.warning('Bad slice for {}, skipping'.format(star_id))
                    continue
            except Exception as e:
                raise e

            # Create group for stamp and add metadata
            try:
                psc_group = stamps[star_id]
            except KeyError:
                logger.debug("Creating new group for star {}".format(star_id))
                psc_group = stamps.create_group(star_id)
                # Stamp metadata
                try:
                    psc_metadata = {
                        'ra': star_row.ra,
                        'dec': star_row.dec,
                        'twomass': star_row.twomass,
                        'vmag': star_row.vmag,
                        'tmag': star_row.tmag,
                        'flags': star_row.flags,
                        'snr': star_row.snr,
                    }
                    for k, v in psc_metadata.items():
                        psc_group.attrs[k] = str(v)
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

    logging.info("Getting Target PSC and subtracting bias")
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
            logging.warning("Skipping frame {}".format(frame_index))

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
                logging.info("Skipping PICID {}, low snr {:.02f}".format(source_index, snr))
                continue
        except KeyError as e:
            logging.debug("No source in table: {}".format(picid))
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
                      aperture_size=4,
                      separate_green=False,
                      subtract_back=False,
                      plot_apertures=False,
                      ):
    """Perform differential aperture photometry on the given PSCs.

    `psc0` and `psc1` are Postage Stamp Cubes (PSC) of N frames x M
    pixels, where M = width x height of the stamp and is assumed to be
    square.

    For each N frame, an aperture is placed around the source in `psc0`
    and the corresponding pixel location in `psc1`. This aperture cutout
    is then split on color channels and for each channel the sum of
    the target, the sum of the reference, and the difference is given.

    Args:
        psc0 (`numpy.array`): An NxM cube of source postage stamps.
        psc1 (`numpy.array`): An NxM cube to be used as the comparison.
        image_times (list(`datetime`)): A list of `datetime.datetime` objects to
            be used for an index.
        aperture_size (int): An aperture around the source that is used
            for photometry, default 4 pixels.
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

    try:
        single_frame = psc0[0].reshape(stamp_side, stamp_side)

        rgb_stamp_masks = helpers.get_rgb_masks(
            single_frame,
            force_new=True,
            separate_green=separate_green
        )
    except ValueError:
        pass

    apertures = list()
    diff = list()
    for frame_idx, image_time in zip(range(num_frames), image_times):

        # Get target and reference stamp for this frame
        t0 = psc0[frame_idx].reshape(stamp_side, stamp_side)
        i0 = psc1[frame_idx].reshape(stamp_side, stamp_side)

        # NOTE: Bad "centroiding" here
        y_pos, x_pos = np.argwhere(t0 == t0.max())[0]
        aperture_position = (x_pos, y_pos)

        for color, mask in rgb_stamp_masks.items():

            # Get color mask data from target and reference
            t1 = np.ma.array(t0, mask=~mask)
            i1 = np.ma.array(i0, mask=~mask)

            # Make apertures
            try:
                t2 = Cutout2D(t1, aperture_position, aperture_size, mode='strict')
                i2 = Cutout2D(i1, aperture_position, aperture_size, mode='strict')
            except (PartialOverlapError, NoOverlapError) as e:
                print(aperture_position, e)
                continue
            except Exception as e:
                print(e)
                continue
                
            t3 = t2.data
            i3 = i2.data
                
            if subtract_back:
                #logging.info("Performing sky background subtraction within aperture")
                #annulus = aperture.RectangularAnnulus(aperture_position, aperture_size + 2, aperture_size * 2, aperture_size + 4)
                #back = annulus.do_photometry(t1.reshape(14, 14), method='center')[0][0]
                #
                #if color == 'g':
                #    pixel_area = annulus.area() / 2
                #else:
                #    pixel_area = annulus.area() / 4
                #    
                #avg_back = back / pixel_area
                mean, median, std = sigma_clipped_stats(t3)
                logging.info("Average sky background for {}: {:5.2f}: {:5.2f}".format(color, mean, t3.sum()))
                
                t3 = t3 - mean
                logging.info(t3.sum())

            t_sum = t3.sum()
            i_sum = i3.sum()

            aps = [t3, i3]
            
            if plot_apertures:
                if subtract_back:
                    apertures.append([t1, i1, annulus])

            diff.append({
                'color': color,
                'target': t_sum,
                'reference': i_sum,
                'obstime': image_time,
            })

    # Light-curve dataframe
    lc0 = pd.DataFrame(diff).set_index(['obstime'])

    if plot_apertures:
        fig = plot.make_apertures_plot(apertures)
        return lc0, fig
    else:
        return lc0


def get_imag(x, t=1):
    return -2.5 * np.log10(x / t)
