import os
import shutil
import subprocess

from warnings import warn
from collections import namedtuple
from glob import glob

import h5py
import numpy as np
import pandas as pd

from scipy import linalg
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.nddata.utils import Cutout2D, PartialOverlapError, NoOverlapError
from astropy.time import Time

from matplotlib import gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from tqdm import tqdm

from dateutil.parser import parse as date_parse

from piaa.utils import helpers

import logging

def normalize(cube):
    return (cube.T / cube.sum(1)).T


def lookup_point_sources(fits_files,
                         image_num=0,
                         catalog_match=True,
                         use_sextractor=True,
                         sextractor_params=None,
                         use_tess_catalog=False,
                         wcs=None,
                         force_new=False,
                         **kwargs
                         ):
    """ Extract point sources from image

    Args:
        image_num (int, optional): Frame number of observation from which to
            extract images
        sextractor_params (dict, optional): Parameters for sextractor,
            defaults to settings contained in the `panoptes.sex` file
        force_new (bool, optional): Force a new catalog to be created,
            defaults to False

    Raises:
        error.InvalidSystemCommand: Description
    """
    if catalog_match or use_tess_catalog:
        assert wcs is not None and wcs.is_celestial, logging.warning("Need a valid WCS")
        
    if use_sextractor:
        # Write the sextractor catalog to a file
        source_file = os.path.join(
            os.environ['PANDIR'], 
            'psc', 
            'point_sources_{}.cat'.format(fits_files[image_num].replace('/', '_'))
        )
        logging.info("Point source catalog: {}".format(source_file))

        if not os.path.exists(source_file) or force_new:
            logging.info("No catalog found, building from sextractor")
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

            logging.info("Running sextractor...")
            cmd = [sextractor, *sextractor_params, fits_files[image_num]]
            logging.info(cmd)
            completed_proc = subprocess.run(cmd, stdout=subprocess.PIPE)

        # Read catalog
        point_sources = Table.read(source_file, format='ascii.sextractor')

        # Remove the point sources that sextractor has flagged
        #if 'FLAGS' in point_sources.keys():
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

    if use_tess_catalog:
        wcs_footprint = wcs.calc_footprint()
        logging.info("WCS footprint: {}".format(wcs_footprint))
                                                                              
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

    if catalog_match:
        point_sources = get_catalog_match(point_sources, wcs, **kwargs)

    return point_sources


def get_catalog_match(point_sources, wcs, table='full_catalog'):
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

    # Change the index to the picid
    point_sources.set_index('id', inplace=True)
    
    return point_sources


def create_stamp_slices(
        stamp_fn,
        fits_files, 
        point_sources, 
        stamp_size=(14, 14), 
        force_new=False,
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

    num_sources = len(point_sources)
    num_frames = len(fits_files)
    
    stamps_fn = os.path.join(
        os.environ['PANDIR'], 
        'psc', 
        stamp_fn.replace('/', '_') + '.hdf5'
    )
    if force_new is False:
        try:
            stamps = h5py.File(stamps_fn)
            return stamps_fn
        except FileNotFoundError:
            pass
        
    stamps = h5py.File(stamps_fn, 'a')

    image_times = np.array([Time(date_parse(fits.getval(fn, 'DATE-OBS'))).mjd for fn in fits_files])
    airmass = np.array([fits.getval(fn, 'AIRMASS') for fn in fits_files])
    
    stamps.attrs['image_times'] = image_times
    stamps.attrs['airmass'] = airmass
    
    for i, fn in tqdm(enumerate(fits_files), total=num_frames, desc='Looping files'):
        # Get stamp data.
        with fits.open(fn) as hdu:
            hdu_idx = 0
            if fn.endswith('.fz'):
                logging.info("Using compressed FITS")
                hdu_idx = 1

            wcs = WCS(hdu[hdu_idx].header)
            d0 = hdu[hdu_idx].data

        for star_row in tqdm(point_sources.itertuples(), total=len(point_sources), leave=False, desc='Sources'):
            star_id = str(star_row.Index)

            if star_id in stamps and np.array(stamps[star_id]['data'][i]).sum() > 0:
                continue

            star_pos = wcs.all_world2pix(star_row.ra, star_row.dec, 0)

            # Get stamp data. If problem, mark for skipping in future.
            try:
                # This handles the RGGB pattern
                slice0 = helpers.get_stamp_slice(star_pos[0], star_pos[1], stamp_size=stamp_size)
                d1 = d0[slice0].flatten()

                if len(d1) == 0:
                    logging.warning('Bad slice for {}, skipping'.format(star_id))
                    continue
            except Exception as e:
                raise e

            # Create group for stamp and add metadata
            try:
                psc_group = stamps[star_id]
            except KeyError:
                logging.debug("Creating new group for star {}".format(star_id))
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
                        logging.warning(e)
                        errors[str(e)] = True

            # Set the data for the stamp. Create PSC dataset if needed.
            try:
                # Assign stamp values
                psc_group['data'][i] = d1
            except KeyError:
                logging.debug("Creating new PSC dataset for {}".format(star_id))
                psc_size = (num_frames, len(d1))

                # Create the dataset
                stamp_dset = psc_group.create_dataset('data', psc_size, dtype='u2', chunks=True)

                # Assign the data
                stamp_dset[i] = d1
            except TypeError as e:
            # Sets the metadata. Create metadata dataset if needed.
                key = str(e) + star_id
                if key not in errors:
                    logging.info(e)
                    errors[key] = True

            try:
                psc_group['original_position'][i] = (star_row.x, star_row.y)
            except KeyError:
                logging.debug("Creating new metadata dataset for {}".format(star_id))
                metadata_size = (num_frames, 2)

                # Create the dataset
                metadata_dset = psc_group.create_dataset(
                        'original_position', metadata_size, dtype='u2', chunks=True)

                # Assign the data
                metadata_dset[i] = (star_row.x, star_row.y)

            stamps.flush()

        if errors:
            logging.warning(errors)

            
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
    out_fn=None, 
    camera_bias=2048,
    num_refs=100, 
    snr_limit=10,
    show_progress=True,
    *args, **kwargs):
    """ Get all variances for given target

    Args:
        stamps(np.array): Collection of stamps with axes: frame, PIC, pixels
        i(int): Index of target PIC
    """
    try:
        return pd.read_csv(out_fn, index_col=[0])
    except Exception:
        pass
    
    num_sources = len(stamps)

    data = dict()

    psc0 = get_psc(picid, stamps, **kwargs) - camera_bias
    num_frames = psc0.shape[0]

    # Normalize
    logging.info("Normalizing target for {} frames".format(num_frames))
    frames = []
    normalized_psc0 = np.zeros_like(psc0, dtype='f4')
    for frame_index in range(num_frames):
        try:
            if psc0[frame_index].sum() > 0.:
                normalized_psc0[frame_index] = psc0[frame_index] / psc0[frame_index].sum()
                frames.append(frame_index)
            else:
                logging.info("Sum for target frame {} is 0".format(frame_index))
        except RuntimeWarning:
            warn("Skipping frame {}".format(frame_index))

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
                continue
        except KeyError:
            pass

        try:
            psc1 = get_psc(source_index, stamps, **kwargs) - camera_bias
        except Exception:
            continue

        normalized_psc1 = np.zeros_like(psc1, dtype='f4')

        # Normalize
        for frame_index in frames:
            if psc1[frame_index].sum() > 0.:
                normalized_psc1[frame_index] = psc1[frame_index] / psc1[frame_index].sum()

        # Store in the grid
        try:
            v = ((normalized_psc0 - normalized_psc1) ** 2).sum()
            data[source_index] = v
        except ValueError as e:
            logging.info("Skipping invalid stamp for source {}: {}".format(source_index, e))

    df0 = pd.DataFrame(
            {'v': list(data.values())}, 
            index=list(data.keys())).sort_values(by='v')
    
    if out_fn:
        df0[:num_refs].to_csv(out_fn)

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

def differential_photometry(psc0, 
                            psc1, 
                            image_times, 
                            aperture_size=4,
                            separate_green=False):
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
            
    Returns:
        `pandas.DataFrame`: A dataframe with `color`, `target`, `reference`, and
            `rel_flux` columsn. 
    
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
    
    diff = list()
    for frame_idx, image_time in zip(range(num_frames), image_times):

        # Get target and reference stamp for this frame
        t0 = psc0[frame_idx].reshape(stamp_side, stamp_side)
        i0 = psc1[frame_idx].reshape(stamp_side, stamp_side)

        # NOTE: Bad "centroiding" here
        try:
            y_pos, x_pos = np.argwhere(t0 == t0.max())[0]
            aperture_position = (x_pos, y_pos)
        except IndexError:
            print("No star position: ", frame_idx, slice0, star_pos_x, star_pos_y)
            continue

        color_flux = dict()
        for color, mask in zip('rgcb', rgb_stamp_masks):

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

            t_sum = t2.data.sum()
            i_sum = int(i2.data.sum())

            diff.append({
                'color': color,
                'target': t_sum,
                'reference': i_sum,
                'rel_flux': t_sum / i_sum,
                'obstime': image_time,
            }) 
            
    # Light-curve dataframe
    lc0 = pd.DataFrame(diff).set_index(['obstime'])
        
    return lc0


def plot_lightcurve(lc0, model_flux=None, transit_info=None, **kwargs):
    """Plot the lightcurve
    
    Args:
        lc0 (`pandas.DataFrame`): The dataframe with ligthcurve info. See
            `differential_photometry` for details.
    """
    fig = Figure()
    FigureCanvas(fig)

    fig.set_size_inches(12, 9)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) 

    ##### Lightcurve Plot #####

    ax1 = fig.add_subplot(gs[0])

    # Raw data values
    ax1.plot(lc0.index, lc0.rel_flux, marker='o', ls='', label='images')

    # Transit model
    if model_flux is not None:
        ax1.plot(lc0.index, model_flux, label='Model transit')

    # Transit lines
    if transit_info is not None:
        midpoint, ingress, egress = transit_info
        ax1.axvline(midpoint, ls='-.', c='g', alpha=0.5)
        ax1.axvline(ingress, ls='--', c='k', alpha=0.5)
        ax1.axvline(egress, ls='--', c='k', alpha=0.5)

    # Unity
    ax1.axhline(1., ls='--', c='k', alpha=0.5)
    ax1.legend(fontsize=16)

    ax1.set_ylim([.96, 1.04])
    
    if 'title' in kwargs:
        ax1.set_title("{}".format(kwargs.get('title')), fontsize=18, y=1.02)

    ##### Residuals Plot #####
    if model_flux is not None:
        ax2 = fig.add_subplot(gs[1])

        residual = lc0.rel_flux - model_flux
        ax2.plot(residual, ls='', marker='o', label='Model {:.04f}'.format(residual.std()))

        ax2.axhline(0, ls='--', alpha=0.5)
        ax2.set_title('Model residual')

    fig.tight_layout()
    
    return fig