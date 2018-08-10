#!/usr/bin/env python3

import os
import argparse
import h5py
import numpy as np
from glob import glob
from getpass import getpass

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.console import ProgressBar

from pong.utils.storage import get_observation_blobs, unpack_blob
from piaa.utils import pipeline
from pocs.utils import current_time
from pocs.utils.images import fits as fits_utils

import logging


def main(sequence, stamp_size=(14,14), snr_limit=10):
    logging.info("Begin sequence: {}".format(sequence))
    start_time = current_time()
    
    # Download FITS files
    fits_blobs = get_observation_blobs(sequence)
    logging.info("Number of images in sequence: {}".format(len(fits_blobs)))
    
    if len(fits_blobs) < 10:
        logging.warning("Not enough images, exiting")
        return
    
    data_dir = os.path.join(
        os.environ['PANDIR'],
        'images', 'fields',
        sequence
    )
    
    # Download all the FITS files from a bucket
    logging.info("Getting files from storage bucket")
    fits_files = list()
    if fits_blobs:
        with ProgressBar(len(fits_blobs)) as bar:
            for i, blob in enumerate(fits_blobs):
                fits_fn = unpack_blob(blob, save_dir=data_dir)
                fits_files.append(fits_fn)
                bar.update(i)

    fits_files = fits_files
    num_frames = len(fits_files)
    
    # Plate-solve all the images - safe to run again
    logging.info("Plate-solving all images")
    solved_files = list()
    with ProgressBar(len(fits_files)) as bar:
        for i, fn in enumerate(fits_files):
            try:
                fits_utils.get_solve_field(fn, timeout=90)
                solved_files.append(fn)
            except Exception as e:
                print("Can't solve file {} {}".format(fn, e))
                continue

    solved_files = solved_files
    
    logging.info("Looking up stars in field")
    wcs = WCS(solved_files[0])
    # Lookup point sources
    # You need to set the env variable for the password for TESS catalog DB (ask Wilfred)
    # os.environ['PGPASSWORD'] = 'sup3rs3cr3t'
    point_sources = pipeline.lookup_point_sources(
        solved_files, 
        wcs=wcs, 
        force_new=True
    )
    
    logging.info("Sources found: {}".format(len(point_sources)))

    # Create stamps
    stamps_fn = pipeline.create_stamp_slices(
        sequence,
        solved_files,
        point_sources[point_sources.snr >= float(snr_limit)],
        stamp_size=stamp_size
    )
    
    if stamps_fn:
        logging.info("Stamps file created: {}".format(stamps_fn))
    
    end_time = current_time()
    logging.info("Total time: {:.02f} seconds".format((end_time - start_time).sec))
    
    return stamps_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a PSC for each detected source.")
    parser.add_argument('--sequence', required=True, type=str, help="Observation sequence.")
    parser.add_argument('--stamp-size', default=14, help="Square stamp size")
    parser.add_argument('--snr-limit', default=10, help="Detected SNR limit for creating stamps.")
    parser.add_argument('--log-level', default='debug', help="Log level")
    
    if 'PGPASSWORD' not in os.environ:
        os.environ['PGPASSWORD'] = getpass(prompt='Catalog DB Password: ')

    args = parser.parse_args()
    
    try:
        log_level = getattr(logging, upper(args.log_level))
    except AttributeError:
        log_level = logging.DEBUG
    finally:
        logging.getLogger().setLevel(log_level)
    
    args.stamp_size = (args.stamp_size, args.stamp_size)

    stamps_fn = main(**vars(args))
    if stamps_fn:
        print("PSC file created: {}".format(stamps_fn))

