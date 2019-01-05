#!/usr/bin/env python3

# # Postage Stamp Cube (PSC) Creation
#
# For each of the sources that were detected we want to slice out the Postage
# Stamp Cube (PSC). The _x_ and _y_ values detected from `sextractor` are used
# to located the star, however this position is then adjusted such that every
# stamp in the cube is aligned and has a red-pixel in the (0,0) position.
# The _x,y_ values for each frame should lie within the center superpixel on the
# corresponding stamp.
#
# The stamps contain the raw data, i.e. no bias removed or any other processing.
#
# Each stamp is saved along with the timestamp and picid inside a csv file, one file per PSC.

import os
from contextlib import suppress
import concurrent.futures
from itertools import zip_longest

import pandas as pd

from tqdm import tqdm

from astropy.io import fits

from piaa.utils import pipeline
from piaa.utils import helpers

from pocs.utils import current_time
from pocs.utils.logger import get_root_logger

import logging
logger = get_root_logger()
logger.setLevel(logging.INFO)


def make_psc(make_params):
    """ Actually do the work of making the PSC. """
    target_group = make_params[0]
    params = make_params[1]

    picid = target_group[0]
    target_table = target_group[1]

    observation_dir = params['observation_dir']
    output_dir = os.path.join(params['output_dir'], str(picid), observation_dir)
    stamp_size = (params['stamp_size'], params['stamp_size'])
    force = params['force']

    os.makedirs(output_dir, exist_ok=True)

    psc_fn = os.path.join(output_dir, 'psc.csv')

    if force:
        with suppress(FileNotFoundError):
            os.remove(psc_fn)

    if not os.path.exists(psc_fn):
        stamps = list()
        for idx, row in target_table.iterrows():

            # Get the data for the entire frame
            data = fits.getdata(row.file)

            # Get the stamp for the target
            try:
                target_slice = helpers.get_stamp_slice(row.x, row.y,
                                                       stamp_size=stamp_size,
                                                       ignore_superpixel=False
                                                       )
                # Get data and flatten
                stamps.append(data[target_slice].flatten())
            except Exception as e:
                logger.warning(f'Problem getting target stamp slice: {e}')


        df0 = pd.DataFrame(stamps, index=target_table.index)
        logger.debug(f'{picid} PSC shape {df0.shape}')
        df0.to_csv(psc_fn)

    return picid


def main(base_dir=None,
         output_dir=None,
         stamp_size=10,
         picid=None,
         force=False,
         num_workers=8,
         chunk_size=12
         ):

    fields_dir = os.path.join(os.environ['PANDIR'], 'images', 'fields')

    # Get the sources from the stored file.
    source_filename = os.path.join(fields_dir, base_dir, f'point-sources-filtered.csv.bz2')

    # Check for existence of file otherwise `lookup_sources_for_observation` will try to create.
    if not os.path.isfile(source_filename):
        raise UserWarning(f'Please do a source detection and filtering first.')

    # Load the sources from the file.
    sources = pipeline.lookup_sources_for_observation(filename=source_filename)
    sources.set_index(['picid'], append=True, inplace=True)

    # Used for progress display.
    num_sources = len(list(sources.index.levels[1].unique()))

    # Add full path to filename in table.
    sources.file = sources.file.apply(lambda fn: os.path.join(fields_dir, base_dir, fn))

    if picid:
        print(f"Creating stamp for {picid}")
        sources = sources.query(f'picid == {picid}')

        if not len(sources):
            print(f"{picid} does not exist, exiting")
            return
    else:
        print(f'Building PSC for {num_sources} sources')

    start_time = current_time()

    call_params = {
        'observation_dir': base_dir,
        'output_dir': output_dir,
        'force': force,
        'stamp_size': stamp_size,
    }

    print(f'Starting at {start_time}')

    # Run everything in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        grouped_sources = sources.groupby('picid')

        params = zip_longest(grouped_sources, [], fillvalue=call_params)

        picids = list(tqdm(executor.map(make_psc, params, chunksize=chunk_size),
                           total=len(grouped_sources)))
        print(f'Created {len(picids)} PSCs')

    end_time = current_time()
    print(f'Ending at {end_time}')
    total_time = (end_time - start_time).sec
    print(f'Total: {total_time:.02f} seconds')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Create a PSC for each detected source.")
    parser.add_argument('--directory', dest='base_dir', default=None, type=str,
                        help="Directory containing observation images.")
    parser.add_argument('--output-dir', default='/var/panoptes/processed', type=str,
                        help=("All artifacts are processed and placed in this directory. "
                              "A subdirectory will be created for each PICID if it does not "
                              "exist and a directory corresponding to the sequence id is made for "
                              "this observation inside the PICID dir. Default $PANDIR/processed/."
                              ))
    parser.add_argument('--stamp-size', type=int, default=10, help="Square stamp size")
    parser.add_argument('--picid', default=None, type=str, help="Create PSC only for given PICID")
    parser.add_argument('--num-workers', default=None, type=int, help="Number of workers to use")
    parser.add_argument('--chunk-size', default=1, type=int, help="Chunks per worker")
    parser.add_argument('--force', action='store_true', default=False,
                        help="Force creation (deletes existing files)")

    args = parser.parse_args()

    print(f'Using {args.num_workers} workers with {args.chunk_size} chunks')
    main(**vars(args))
    print('Finished creating PSC files')
