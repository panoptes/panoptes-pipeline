#!/usr/bin/env python3

import os
import argparse
import h5py
import numpy as np
import concurrent.futures
from glob import glob
from itertools import zip_longest
from getpass import getpass

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D, NoOverlapError, PartialOverlapError
from photutils import make_source_mask

from piaa.utils.helpers import get_stars, get_observation_blobs, unpack_blob
from piaa import pipeline

from pocs.utils import current_time
from pocs.utils.images.fits import get_solve_field, fpack

start_time = current_time()
source_mask = None
stamp_size = 7


def _print(*args):
    t = current_time()
    dt = (t - start_time).sec
    print("{:08.02f}".format(dt), *args)


bias_node = pipeline.BiasSubtract()
dark_node = pipeline.DarkCurrentSubtract(20)
mask_node = pipeline.MaskBadPixels()
back_node = pipeline.BackgroundSubtract()
pipe = pipeline.PanPipeline([bias_node, mask_node])


def calibrate_image(input_args):
    fits_fn, working_dir = input_args

    with fits.open(fits_fn) as hdu:
        h0 = hdu[0].header
        d0 = hdu[0].data

    image_id = h0['IMAGEID']
    image_path = os.path.join(working_dir, image_id + '.npy')
    _print("Starting", image_id)

    if not os.path.exists(image_path):
        d1 = pipe.run(d0, source_mask=source_mask)
        np.save(image_path, d1.filled(0).astype(np.int16))

    _print("Done", image_id)
    return image_path


def extract_stamp(image_file, positions):
    x, y = positions

    # Set a default
    stamp = np.zeros((stamp_size, stamp_size))

    # Load (via memmap) the stamp
    try:
        d0 = np.load(image_file, mmap_mode='r')
    except ValueError as e:
        _print("Problem loading file: {} {}".format(image_file, e))

    # Get cutouts
    try:
        cutout = Cutout2D(d0, (x, y), stamp_size, mode='strict').data
    except (NoOverlapError, PartialOverlapError):
        pass
    else:
        stamp = cutout

    return stamp.flatten()


def main(seq_id=None, image_dir=None, camera_id=None):
    global source_mask
    _print("Begin sequence", seq_id)

    # Make working dir
    _print("Making working directory")
    working_dir = os.path.join('/var/panoptes/processed', seq_id.replace('/', '_'))
    psc_dir = os.path.join(working_dir, 'psc_{}'.format(stamp_size))
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(psc_dir, exist_ok=True)
    
    # Make HD5F file
    h5 = h5py.File('{}/stamps.hdf5'.format(working_dir), 'w')

    # Get the fits files
    _print("Getting blobs")
    blobs = get_observation_blobs(seq_id)
    
    _print("Getting fits files")
    fits_files = [unpack_blob(b) for b in blobs]
    
    _print("Solving fits files")
    solve_info = [get_solve_field(f, solve_opts=[
        '--guess-scale',
        '--cpulimit', '90',
        '--no-verify',
        '--no-plots',
        '--crpix-center',
        '--match', 'none',
        '--corr', 'none',
        '--wcs', 'none',
        '--downsample', '4',
        '--continue', 
        '-t', '3', 
        '-q', '0.01', 
        '-V', f
    ]) for f in fits_files]
    
    _print("Calibrating data")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        calibrate_args = list(zip_longest(fits_files, [], fillvalue=working_dir))
        calibrated_list = list(executor.map(calibrate_image, calibrate_args))
        
    _print("Getting all WCS")
    wcs_list = [WCS(f, naxis=0) for f in fits_files]
    print(wcs_list[0])

    _print("Looking up stars")
    wcs_footprint = wcs_list[0].calc_footprint()
    ra_max = max(wcs_footprint[:, 0])
    ra_min = min(wcs_footprint[:, 0])
    dec_max = max(wcs_footprint[:, 1])
    dec_min = min(wcs_footprint[:, 1])
    _print("RA: {:.03f} - {:.03f} \t Dec: {:.03f} - {:.03f}".format(ra_min, ra_max, dec_min, dec_max))
    star_table = get_stars(ra_min, ra_max, dec_min, dec_max, cursor_only=False)
    _print(len(star_table), "stars in field")

    csv_file = os.path.join(working_dir, 'pscs.csv')
    
    _print("Extracting stamps from stars (size={})".format(stamp_size))
    
    h5_dset = h5.create_dataset("stamps", (len(star_table), len(fits_files), stamp_size*stamp_size), chunks=(1, len(fits_files), stamp_size*stamp_size))
    
    #with open(csv_file, 'w') as csv_fn:
        #writer = csv.writer(csv_fn)
        
    for j, fits_fn in enumerate(fits_files):
        _print(fits_fn)
        with fits.open(fits_fn) as hdu:
            w = WCS(hdu[0].header)
            h = hdu[0].header

        for i, star in enumerate(star_table):
            picid, ra, dec, tmag, e_tmag, twomass = star

            positions = w.all_world2pix(ra, dec, 0)

            psc = extract_stamp(calibrated_list[j], positions)

            # Save the PSC (don't save zero sum or irregularly shaped)
            try:
                if psc.sum() > 0:
                    #date_obs = h['DATE-OBS']
                    #star_info = [picid, date_obs]
                    #star_info.extend(psc)
                    #writer.writerow(star_info)
                    h5_dset[i, j] = psc.flatten()
            except ValueError as e:
                _print(e)
                pass
        
    _print("Removing FITS files")
    for f in fits_files:
        try:
            os.remove(f)
        except Exception as e:
            pass

    end_time = current_time()
    _print("Total time:", (end_time - start_time).sec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default='/var/panoptes/fits_files',
                        help="Image directory containing FITS files")
    parser.add_argument('--seq_id', required=True, type=str,
                        help="Image sequence of Observation")
    parser.add_argument('--camera_id', required=False, type=str,
                        help="ID of camera")

    if 'PGPASSWORD' not in os.environ:
        os.environ['PGPASSWORD'] = getpass(prompt='Catalog DB Password: ')

    args = parser.parse_args()

    main(**vars(args))

