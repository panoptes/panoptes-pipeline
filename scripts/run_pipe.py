#!/usr/bin/env python3

import os
import argparse
import numpy as np
import concurrent.futures
import h5py
from glob import glob
from itertools import zip_longest
from getpass import getpass

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D, NoOverlapError, PartialOverlapError
from photutils import make_source_mask

from piaa.utils.helpers import get_stars
from piaa import pipeline

from pocs.utils import current_time

start_time = current_time()
source_mask = None


def _print(*args):
    t = current_time()
    dt = (t - start_time).sec
    print("{:08.02f}".format(dt), *args)


bias_node = pipeline.BiasSubtract()
dark_node = pipeline.DarkCurrentSubtract(20)
mask_node = pipeline.MaskBadPixels()
back_node = pipeline.BackgroundSubtract()
pipe = pipeline.PanPipeline([bias_node, dark_node, mask_node, back_node])


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
    params, stamp_size = input_args
    positions, image_file = params
    x, y = positions

    # Set a default
    stamp = np.zeros((stamp_size, stamp_size))

    # Load (via memmap) the stamp
    d0 = np.load(image_file, mmap_mode='r')

    # Get cutouts
    try:
        cutout = Cutout2D(d0, (x, y), stamp_size, mode='strict').data
    except (NoOverlapError, PartialOverlapError):
        pass
    else:
        stamp = cutout

    return stamp.flatten()


def main(seq_id=None, image_dir=None):
    global source_mask
    _print("Begin sequence", seq_id)

    # Make working dir
    _print("Making working directory")
    working_dir = os.path.join('/var/panoptes/processed', seq_id.replace('/', '_'))
    psc_dir = os.path.join(working_dir, 'psc')
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(psc_dir, exist_ok=True)

    fits_files = sorted(glob('*.fz'.format(image_dir)))

    _print("Making source mask")
    with fits.open(fits_files[0]) as hdu:
        source_mask = make_source_mask(
            hdu[0].data, snr=3, npixels=3, sigclip_sigma=3, sigclip_iters=5, dilate_size=12)
        wcs = WCS(hdu[0].header)

    hdf5_path = os.path.join(working_dir, image_dir + '.hdf5')

    if not os.path.exists(hdf5_path):
        _print("Calibrating data")
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            calibrate_args = list(zip_longest(fits_files, [], fillvalue=working_dir))
            calibrated_list = list(executor.map(calibrate_image, calibrate_args))

        _print("Create empty HDF5 data cube")
        hdf5 = h5py.File(hdf5_path)
        cube_dset = hdf5.create_dataset(
            'cube', (len(fits_files), source_mask.shape[0], source_mask.shape[1]))
        for i, f in enumerate(fits_files):
            cube_dset[i] = fits.getdata(f)

        _print("Saving calibrated data to HDF5 cube")
        for i, f in enumerate(calibrated_list):
            d0 = np.load(f)
            cube_dset[i] = d0
            os.remove(f)

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

# stamp_size = 7

#  bad_positions = list()
# _print("Looping files to extract stars")
# for f_idx, f in enumerate(calibrated_list):
#     _print("Extracting from", f)
#     # Get star positions for this file
#     star_positions = np.array(wcs_list[f_idx].wcs_world2pix(star_table['ra'], star_table['dec'], 0)).T

#     d0 = np.load(f, mmap_mode='r')

#     # Get stamp for each star
#     for pos_idx,  position in enumerate(star_positions):
#         if np.any(position < 0) or pos_idx in bad_positions:
#             continue

#         psc_path = os.path.join(working_dir, 'psc', '{}.npy'.format(star_table[pos_idx]['id']))

#         try:
#             stamp = Cutout2D(d0, position, stamp_size, mode='strict').data.flatten()
#         except (NoOverlapError, PartialOverlapError):
#             bad_positions.append(pos_idx)
#         else:
#             # Load existing cube, append, save
#             try:
#                 # Load existing cube
#                 psc = np.load(psc_path, mmap_mode='r+')
#             except FileNotFoundError:
#                 # Create a cube of zeros
#                 np.save(psc_path, np.zeros((len(calibrated_list), stamp_size*stamp_size)))
#                 # Load new cube
#                 psc = np.load(psc_path, mmap_mode='r+')
#             finally:
#                 psc[f_idx] = stamp
#                 psc.flush()

    _print("Getting star positions")

    # For each WCS, get the X,Y position and then add the star ID to same array
    star_positions = [
        np.insert(
            np.array(w.wcs_world2pix(star_table['ra'], star_table['dec'], 0)).T,
            2,
            star_table['id'],
            axis=1
        )
        for w in wcs_list
    ]

    np.save(os.path.join(working_dir, 'star_positions'), star_positions)


#    _print("Extracting stamps from stars (size={})".format(stamp_size))
#    for i, star in enumerate(star_table):
#        if i % 2501 == 0:
#            _print(i, "{:%}".format(i / len(star_table)))
#
#        psc_path = os.path.join(psc_dir, str(star['id']))
#
#        if os.path.exists(psc_path):
#            _print("Skipping", psc_path)
#            continue
#
#        # Lookup star xy positions for each image
#        stamp_args = [(w.wcs_world2pix(star['ra'], star['dec'], 0), image_file) for w, image_file in processed_files]
#
#        # Spin off a process for each file for this star
#        with concurrent.futures.ThreadPoolExecutor() as executor:
#            stamp_args = list(zip_longest(stamp_args, [], fillvalue=stamp_size))
#            psc = np.array(list(executor.map(extract_stamp, stamp_args)))
#
#        # Save the PSC (don't save zero sum or irregularly shaped)
#        try:
#            if psc.sum() > 0:
#                #_print("Saving cube for ", star['id'])
#                np.save(psc_path, psc)
#        except ValueError as e:
#            _print(e)
#            pass

    end_time = current_time()
    _print("Total time:", (end_time - start_time).sec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True, type=str,
                        default='/var/panoptes/images',
                        help="Image directory containing FITS files")
    parser.add_argument('--seq_id', required=True, type=str,
                        help="Image sequence of Observation")

    if 'PGPASSWORD' not in os.environ:
        os.environ['PGPASSWORD'] = getpass(prompt='Catalog DB Password: ')

    args = parser.parse_args()

    main(**args)
