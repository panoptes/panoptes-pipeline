#!/usr/bin/env python3

# # Postage Stamp Cube (PSC) Creation
# 
# For each of the sources that were detected we want to slice out the Postage Stamp Cube (PSC). The _x_ and _y_ values detected from `sextractor` are used to located the star, however this position is then adjusted such that every stamp in the cube is aligned and has a red-pixel in the (0,0) position. The _x,y_ values for each frame should lie within the center superpixel on the corresponding stamp.
# 
# The stamps contain the raw data, i.e. no bias removed or any other processing. 
# 
# Each stamp is saved along with the timestamp and picid inside a csv file, one file per PSC.

import os
import logging
from contextlib import suppress

import numpy as np
import pandas as pd

from tqdm import tqdm

from astropy.io import fits

from piaa.utils import pipeline
from piaa.utils import helpers

from pocs.utils import current_time


def main(base_dir=None, stamp_size=10):
    
    fields_dir = os.path.join(os.environ['PANDIR'], 'images', 'fields')

    source_filename = os.path.join(base_dir, f'point-sources-filtered.csv.bz2')
    
    assert os.path.isfile(source_filename)

    # Actually get the sources
    sources = pipeline.lookup_sources_for_observation(filename=source_filename).set_index(['picid'], append=True)

    # Make directory for PSC
    os.makedirs(os.path.join(base_dir, 'stamps'), exist_ok=True)

    # Used for progress display
    num_sources = len(list(sources.index.levels[1].unique()))
    print(f'Building PSC for {num_sources} sources')

    for pid, target_table in tqdm(sources.groupby('picid'), total=num_sources, desc="Making PSCs"):
        stamps = list()
        for idx, row in target_table.iterrows():
            date_obs= idx[0]

            # Get the data for the entire frame
            data = fits.getdata(os.path.join(base_dir, row.file)) 

            # Get the stamp for the target
            target_slice = helpers.get_stamp_slice(row.x, row.y, stamp_size=(stamp_size, stamp_size), ignore_superpixel=False)

            # Get data
            stamps.append(data[target_slice].flatten())

        pd.DataFrame(stamps, index=target_table.index).to_csv(os.path.join(base_dir, 'stamps', f'{pid}.csv'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Create a PSC for each detected source.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--directory', default=None, type=str,
                       help="Directory containing observation images.")
    parser.add_argument('--stamp-size', default=10, help="Square stamp size")
    parser.add_argument('--log_level', default='debug', help="Log level")
    parser.add_argument(
        '--log_file', help="Log files, default $PANLOG/create_stamps_<datestamp>.log")

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

    logging.info('*' * 80)
    ################ End Setup logging ##############
    
    assert os.path.isdir(args.directory)

    main(base_dir=args.directory, stamp_size=args.stamp_size)
    print('Finished creating stamps')
