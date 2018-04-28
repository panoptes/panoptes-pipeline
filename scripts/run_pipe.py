#!/usr/bin/env python3

import os
import argparse
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
#pipe = pipeline.PanPipeline([bias_node, dark_node, mask_node, back_node])
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


def extract_stamp(input_args):
    image_file, positions = input_args
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
    
    #fits_files = sorted(glob('{}/*{}*.fits'.format(image_dir, camera_id)))

    _print("Making source mask")
    with fits.open(fits_files[0]) as hdu:
        source_mask = make_source_mask(
            hdu[0].data, snr=3, npixels=3, sigclip_sigma=3, sigclip_iters=5, dilate_size=12)
        wcs = WCS(hdu[0].header)

    _print("Calibrating data")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        calibrate_args = list(zip_longest(fits_files, [], fillvalue=working_dir))
        calibrated_list = list(executor.map(calibrate_image, calibrate_args))

    _print("Looking up stars")
    wcs_footprint = wcs.calc_footprint()
    ra_max = max(wcs_footprint[:, 0])
    ra_min = min(wcs_footprint[:, 0])
    dec_max = max(wcs_footprint[:, 1])
    dec_min = min(wcs_footprint[:, 1])
    star_table = get_stars(ra_min, ra_max, dec_min, dec_max)
    _print(len(star_table), "stars in field")

    _print("Getting all WCS")
    wcs_list = [WCS(f) for f in fits_files]
    
    _print("Extracting stamps from stars (size={})".format(stamp_size))
    for i, star in enumerate(star_table):
        if i % 2500 == 0:
            _print(i, "{:.02%}".format(i / len(star_table)))

        psc_path = os.path.join(psc_dir, str(star['id']) + '.npz')

        if os.path.exists(psc_path):
            #_print("Skipping", psc_path)
            continue

        # Lookup star xy positions for each image
        # Use only the first WCS (see commented out code below)
        #stamp_args = {
        #        image_file:wcs_list[0].all_world2pix(star['ra'], star['dec'], 0)
        #        for image_file in calibrated_list
        #}
    
        stamp_args = {
                image_file:w.all_world2pix(star['ra'], star['dec'], 0)
                for w, image_file in zip(wcs_list, calibrated_list)
        }

        # Spin off a process for each file for this star
        with concurrent.futures.ThreadPoolExecutor(max_workers=10*50) as executor:
            psc = np.array(list(executor.map(extract_stamp, stamp_args.items())))

        # Save the PSC (don't save zero sum or irregularly shaped)
        try:

            if psc.sum() > 0:
                #_print("Saving cube for ", star['id'])
                np.savez_compressed(psc_path, psc=psc, pos=np.array(list(stamp_args.values())))
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

