#!/usr/bin/env python3

# ## Find similar stars
# 
# Because a simple differential photometry doesn't work well we need to find a suitable set of reference stars that undergo the same changes (e.g. airmass, cloud cover, etc) as our target star.
# 
# Ultimately we are concerned about how the flux of our target changes with respect to a suitable set of reference stars and therefore we need to be careful **not** to use flux as a selection parameter for determining the "best" reference stars. That is, we **do not** want to choose reference stars that undergo a reduction of flux in the middle of the observation as this will hide the transit signal in our target.
# 
# We can marginalize across the flux by normalizing each stamp to the total flux in the stamp (see [Normalize](Algorithm.ipynb#normalize) in algorithm description). By doing so we are effectively looking at the shape of the star as it appears on the RGB pixel pattern. This morphology will change slightly from frame to frame so we want to look for reference stars that change similarly to our target star with respect to this morphology.
# 
# By taking the summed squared difference (SSD) between each pixel of the normalized target and reference star we can get a single metric that defines how well the reference star matches the target. Because the SSD is looking at the difference between the target and a reference, a lower metric value for the refernce indicates a better match with the target. The target stamp compared with itself would yield a value of zero.
# 
# We perform the SSD for each frame in the observation and take the sum of all the SSDs for each source as the final metric score to compare against our target. Again, lower scores mean that the reference is more "similar" in a morphological sense: it's shape on the RGB pattern changes similar to that of the target. See [Find Reference Stars](Algorithm.ipynb#find_reference) for details and mathematical description.

# ### Get the ranking for comparison stars
# 
# For each source that was identified above we want to find the most "similar" stars by ranking them according to how the shape of their PSF differs from that of the target. This is done for each frame and the sum across all frames determines the "similarity", with smaller final sums indicating stars that are similar to the target. The target ranked against itself would yield a value of zero.
# 
# By the numbers, this is doing the sum of the summed squared difference (SSD) for each pixel in the stamp for each frame. Importantly, it is doing this comparision on the normalized version of each stamp. The stamp is normalized according to the total sum of the stamp. See [Step 1](Algorithm.ipynb#normalize) below for the Normalization and [Step 2](Algorithm.ipynb#find_references) for the sum of the SSD.

import os
import logging
from contextlib import suppress
import concurrent.futures

import pandas as pd
import numpy as np
from collections import defaultdict

from matplotlib import pyplot as plt

from glob import glob
from tqdm import tqdm

from pocs.utils import current_time

# How many matches to save
SAVE_NUM=500


def get_normalized_psc(stamp_fn, camera_bias=2048):
    """ Reads a postage stamp cube file and returns the normalized version """
    source_table = pd.read_csv(stamp_fn).set_index(['obs_time', 'picid'])
    source_psc = np.array(source_table) - camera_bias

    # Normalize
    normalized_psc = (source_psc.T / source_psc.sum(1)).T
    
    return normalized_psc

def find_similar(stamp_fn, camera_bias=2048):
    """ The worker thread to find the stars """
    
    # Get the picid from the filename
    picid = os.path.splitext(os.path.basename(stamp_fn))[0]
    stamps_dir = os.path.dirname(stamp_fn)
    
    similar_dir = os.path.join(stamps_dir, 'similar')
    
    similar_fn = os.path.normpath(os.path.join(similar_dir, f'{picid}.csv'))
    
    if not os.path.exists(similar_fn):
        # Normalize target PSC
        normalized_target_psc = get_normalized_psc(stamp_fn)

        # Get all the stamp files
        stamp_files = glob(os.path.join(stamps_dir, '*.csv'))

        # Loop through all other stamp files
        vary = dict()
        for comp_stamp_fn in stamp_files:
            ref_picid = os.path.splitext(os.path.basename(comp_stamp_fn))[0]

            normalized_ref_psc = get_normalized_psc(comp_stamp_fn)

            try:
                score = ((normalized_target_psc - normalized_ref_psc)**2).sum()
            except ValueError:
                continue

            vary[ref_picid] = score

        vary_series = pd.Series(vary).sort_values()
        vary_series[:SAVE_NUM].to_csv(similar_fn)
    
    return picid

def main(base_dir,
         picid=None,
         force=False,
         num_workers=8,
         chunk_size=12,
    ):
    print(f'Finding similar stars for observation in {base_dir}')

    if picid:
        print(f'Searching for picid={picid}')
        stamp_files = glob(os.path.join(base_dir, 'stamps', f'{picid}.csv'))
    else:
        stamp_files = glob(os.path.join(base_dir, 'stamps', '*.csv'))
        
    print(f'Found {len(stamp_files)} PSC files')
    
    similar_dir = os.path.join(base_dir, 'stamps', 'similar')
    os.makedirs(similar_dir, exist_ok=True)
    
    if force:
        print(f'Forcing creation, deleting all similar rankings in {similar_dir}')
        for fn in glob(f'{similar_dir}/*.csv'):
            with suppress(FileNotFoundError):
                os.remove(fn)

    start_time = current_time()
    print(f'Starting at {start_time}')

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
        picids = list(tqdm(executor.map(find_similar, stamp_files, chunksize=int(chunk_size)), total=len(stamp_files)))
        print(f'Found similar stars for {len(picids)} sources')
            
    end_time = current_time()
    print(f'Ending at {end_time}')
    total_time = (end_time - start_time).sec
    print(f'Total: {total_time:.02f} seconds')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Find similar stars for each star.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--directory', default=None, type=str,
                       help="Directory containing observation images.")
    parser.add_argument('--picid', default=None, type=str, help="Create PSC only for given PICID")
    parser.add_argument('--num-workers', default=8, help="Number of workers to use")
    parser.add_argument('--chunk-size', default=10, help="Chunks per worker")
    parser.add_argument('--force', action='store_true', default=False, 
                        help="Force creation (deletes existing files)")

    args = parser.parse_args()

    fields_dir = os.path.join(os.environ['PANDIR'], 'images', 'fields')
    base_dir = os.path.join(fields_dir, args.directory)
    
    assert os.path.isdir(base_dir)

    print(f'Using {args.num_workers} workers with {args.chunk_size} chunks')
    main(
        base_dir=args.directory,
        picid=args.picid,
        force=args.force,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size
    )
    print('Finished creating stamps')
    