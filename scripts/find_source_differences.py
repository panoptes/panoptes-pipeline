#!/usr/bin/env python3

import os
import h5py                                   
from glob import glob
from tqdm import tqdm
from contextlib import suppress
import numpy as np

from piaa.utils import pipeline
from pocs.utils import current_time

import logging


def make_2dgauss(width):
    x, y = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,width))

    d = np.sqrt(x**2 + y**2)
    sigma, mu = 0.2, 0.0

    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    
    return g / g.sum()  # Normalize


def main(stamp_file, show_progress=True, *args, **kwargs):
    try:
        stamps = h5py.File(stamp_file)
    except FileNotFoundError:
        logging.warning("File not found: {}".format(stamp_file))
        return
    
    star_iterator = enumerate(list(stamps.keys()))
    
    # Show progress bar
    if show_progress:
        star_iterator = tqdm(star_iterator, total=len(list(stamps.keys())), desc='Looping sources')
        
    gauss = None

    for i, picid in star_iterator:
        flags = stamps[picid].attrs['flags']
        if int(flags):
            logging.debug("Skipping {} - SE flags: {}".format(picid, flags))
            continue

        if float(stamps[picid].attrs['vmag']) > 13:
            logging.debug("Skipping {} - Vmag: {:.02f} > 13".format(
                picid, 
                float(stamps[picid].attrs['vmag']))
            )
            continue
            
        psc0 = np.array(stamps[picid]['data'])
        normalized_psc0 = pipeline.normalize(psc0)
        
        if gauss is None:
            width = int(np.sqrt(psc0.shape[1]))
            gauss = make_2dgauss(width).flatten()
            
        similarity_sum = ((normalized_psc0 - gauss) ** 2).sum()

        # Store in stamps file
        logging.info("Success {}".format(picid))
        stamps[picid].attrs['similarity'] = similarity_sum
        stamps.flush()
 

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Find morphologically similar stars")
    parser.add_argument('--stamp_file', required=True, type=str, help="HDF5 Stamps file")
    parser.add_argument('--log_level', default='debug', help="Log level")
    parser.add_argument('--log_file', help="Log files, default $PANLOG/create_stamps_<datestamp>.log")
    parser.add_argument('--show_progress', default=True, action='store_true', 
                        help="Show progress bars")
    
    args = parser.parse_args()
    
    ################ Setup logging ##############
    log_file = os.path.join(
        os.environ['PANDIR'], 
        'logs', 
        'per-run',
        'find_similar_sources_{}.log'.format(current_time(flatten=True))
    )
    common_log_path = os.path.join(
        os.environ['PANDIR'],
        'logs',
        'find_similar_sources.log'
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
    
    stamps_fn = main(**vars(args))
    if stamps_fn:
        print("Similar sources added to {}".format(stamps_fn))
