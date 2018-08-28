#!/usr/bin/env python3

import os
import argparse
import h5py
import numpy as np
from glob import glob
from getpass import getpass
from contextlib import suppress

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.console import ProgressBar

from pong.utils.storage import get_observation_blobs, download_fits_file
from piaa.utils import pipeline
from pocs.utils import current_time
from pocs.utils.images import fits as fits_utils

import logging


def main(sequence=None, files=None, stamp_size=(14,14), snr_limit=10, *args, **kwargs):
    start_time = current_time()
    
    if files is None and sequence is not None:
        logging.info("Begin sequence: {}".format(sequence))
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
                    fits_fn = download_fits_file(blob, save_dir=data_dir, unpack=False)
                    fits_files.append(fits_fn)
                    bar.update(i)
    elif files is not None:
        fits_files = files
        sequence = fits.getval(fits_files[0], 'SEQID')
    
    num_frames = len(fits_files)
    logging.info("Using sequence id {} with {} frames".format(sequence, num_frames))
    
    # Plate-solve all the images - safe to run again
    logging.info("Plate-solving all images")
    solved_files = list()
    with ProgressBar(len(fits_files)) as bar:
        for i, fn in enumerate(fits_files):
            try:
                fits_utils.get_solve_field(fn, timeout=90, verbose=True)
                solved_files.append(fn)
            except Exception as e:
                print("Can't solve file {} {}".format(fn, e))
                continue

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
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sequence', default=None, type=str, help="Observation sequence.")
    group.add_argument('--directory', default=None, type=str, help="Directory containing observation images.")
    parser.add_argument('--stamp-size', default=14, help="Square stamp size")
    parser.add_argument('--snr_limit', default=10, help="Detected SNR limit for creating stamps.")
    parser.add_argument('--log_level', default='debug', help="Log level")
    parser.add_argument('--log_file', help="Log files, default $PANLOG/create_stamps_<datestamp>.log")
    
    if 'PGPASSWORD' not in os.environ:
        os.environ['PGPASSWORD'] = getpass(prompt='Catalog DB Password: ')

    args = parser.parse_args()
    
    ################ Setup logging ##############
    log_file = os.path.join(
        os.environ['PANDIR'], 
        'logs', 
        'per-run',
        'create_stamps_{}.log'.format(current_time(flatten=True))
    )
    common_log_path = os.path.join(
        os.environ['PANDIR'],
        'logs',
        'create_stamps.log'
    )
    
    if args.log_file:
        log_file = args.log_file
        
    try:
        log_level = getattr(logging, args.log_level.upper())
    except AttributeError:
        log_level = logging.DEBUG
    finally:
        logging.basicConfig(filename=log_file, level=log_level)
        
        with suppress(FileNotFoundError):
            os.remove(common_log_path)
            
        os.symlink(log_file, common_log_path)
        
    logging.info('*'*80)
    ################ End Setup logging ##############
    
    files = None
    if args.directory is not None:
        files = sorted(glob(os.path.join(
            args.directory,
            '*.fits'
        ), recursive=True))
        
        logging.info("Found {} FITS files in {}".format(len(files), args.directory))
    
    args.stamp_size = (args.stamp_size, args.stamp_size)

    stamps_fn = main(**vars(args), files=files)
    if stamps_fn:
        print("PSC file created: {}".format(stamps_fn))

