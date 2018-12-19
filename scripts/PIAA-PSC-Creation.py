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
import concurrent.futures
from glob import glob

import numpy as np
import pandas as pd

from tqdm import tqdm

from astropy.io import fits

from piaa.utils import pipeline
from piaa.utils import helpers

from pocs.utils import current_time


def make_psc(target_group):
    """ Actually do the work of making the PSC. """
    picid = target_group[0]
    target_table = target_group[1]
    
    # Get the observation directory
    base_dir = os.path.dirname(target_table.iloc[0].file)
        
    psc_fn = os.path.join(base_dir, 'stamps', f'{picid}.csv')
    
    if not os.path.exists(psc_fn):
        stamps = list()
        for idx, row in target_table.iterrows():
            date_obs= idx[0]

            # Get the data for the entire frame
            data = fits.getdata(row.file) 
            stamp_size = row.stamp_size

            # Get the stamp for the target
            target_slice = helpers.get_stamp_slice(row.x, row.y, stamp_size=(stamp_size, stamp_size), ignore_superpixel=False)

            # Get data
            stamps.append(data[target_slice].flatten())

        pd.DataFrame(stamps, index=target_table.index).to_csv(os.path.join(base_dir, 'stamps', f'{picid}.csv'))
        
    return picid


def main(base_dir=None,
         stamp_size=10,
         picid=None,
         force=False,
         num_workers=8,
         chunk_size=12
    ):
    
    fields_dir = os.path.join(os.environ['PANDIR'], 'images', 'fields')
    source_filename = os.path.join(base_dir, f'point-sources-filtered.csv.bz2')
    assert os.path.isfile(source_filename)

    # Get the sources
    sources = pipeline.lookup_sources_for_observation(filename=source_filename).set_index(['picid'], append=True)

    # Make directory for PSC
    stamp_dir = os.path.join(base_dir, 'stamps')
    os.makedirs(stamp_dir, exist_ok=True)
    
    if force:
        print(f'Forcing creation, deleting all stamps in {stamp_dir}')
        for fn in glob(f'{stamp_dir}/*.csv'):
            with suppress(FileNotFoundError):
                os.remove(fn)

    # Used for progress display
    num_sources = len(list(sources.index.levels[1].unique()))
    
    # Add the stamp size and base dir to table (silly way to pass args)
    sources['file'] = [os.path.join(base_dir, row.file) for _, row in sources.iterrows()]
    sources['stamp_size'] = stamp_size
    
    if picid:
        print(f"Creating stamp for {picid}")
        sources = sources.query(f'picid == {picid}')
        
        if not len(sources):
            print(f"{picid} does not exist, exiting")
            return
    else:
        print(f'Building PSC for {num_sources} sources')
    
    start_time = current_time()
    
    print(f'Starting at {start_time}')

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for i, picid in enumerate(executor.map(make_psc, sources.groupby('picid'), chunksize=12)):
            print(f'Finished with {picid}: {i}/{num_sources}')
            
    end_time = current_time()
    print(f'Ending at {end_time}')
    total_time = (end_time - start_time).sec
    print(f'Total: {total_time:.02f} seconds')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Create a PSC for each detected source.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--directory', default=None, type=str,
                       help="Directory containing observation images.")
    parser.add_argument('--stamp-size', default=10, help="Square stamp size")
    parser.add_argument('--picid', default=None, type=str, help="Create PSC only for given PICID")
    parser.add_argument('--num-workers', default=8, help="Number of workers to use")
    parser.add_argument('--chunk-size', default=10, help="Chunks per worker")
    parser.add_argument('--force', action='store_true', default=False, 
                        help="Force creation (deletes existing files)")
    parser.add_argument('--log_level', default='debug', help="Log level")
    parser.add_argument(
        '--log_file', help="Log files, default $PANLOG/create_stamps_<datestamp>.log")

    args = parser.parse_args()

    assert os.path.isdir(args.directory)

    print(f'Using {args.num_workers} workers with {args.chunk_size} chunks')
    main(
        base_dir=args.directory,
        stamp_size=args.stamp_size,
        picid=args.picid,
        force=args.force,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size
    )
    print('Finished creating stamps')
