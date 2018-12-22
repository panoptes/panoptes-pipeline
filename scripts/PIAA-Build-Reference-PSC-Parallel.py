#!/usr/bin/env python3

# ## Build Reference PSC

import os
import numpy as np
import pandas as pd
from copy import copy
from itertools import zip_longest
import concurrent.futures

from glob import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from tqdm import tqdm

from astropy.time import Time
from astropy.stats import sigma_clip

from piaa.exoplanets import TransitInfo, get_exoplanet_transit
from piaa.utils import helpers
from piaa.utils import plot
from piaa.utils import pipeline

from pocs.utils import current_time

plt.style.use('bmh')

    
def build_ref(build_params):
    """ Build a reference PSC for the given PSC. """
    
    psc_fn = build_params[0]
    params = build_params[1]
    
    # Load params
    aperture_size = params['aperture_size']
    camera_bias = params['camera_bias']
    num_refs = params['num_refs']
    frame_slice = params['frame_slice']
    table_filter = params['table_filter']
    make_plots = params['make_plots']
    
    # Get working directories.
    psc_dir = os.path.dirname(psc_fn)
    base_dir = os.path.dirname(psc_dir)
    similar_dir = os.path.join(psc_dir, 'similar')
    
    # Get current picid from filename.
    picid = os.path.splitext(os.path.basename(psc_fn))[0]
    
    # Subdirectory to place processed files.
    output_dir = os.path.normpath(os.path.join(base_dir, 'processed', str(picid)))
    os.makedirs(output_dir, exist_ok=True)
    if make_plots:
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Get the list of similar stars. The picid at index 0 is the target star.
    similar_stars = pd.read_csv(
        os.path.join(similar_dir, f'{picid}.csv'),
        names=['picid', 'score']
    )
                                  
    if make_plots:
        try:
            plot_similarity(picid, similar_stars, output_dir)
        except Exception as e:
            logger.error(f"Can't make similarity plot for {picid}")
            logger.error(e)

    # Get the list of picids.
    ref_picid_list = list(similar_stars.picid[:num_refs].values)

    # Build stamp collection from references
    psc_collection = list()
    for i, ref_picid in enumerate(ref_picid_list):
        ref_psc_fn = os.path.join(psc_dir, f'{ref_picid}.csv')

        try:
            full_ref_table = pd.read_csv(ref_psc_fn).set_index(['obstime', 'picid'])
        except KeyError:
            full_ref_table = pd.read_csv(ref_psc_fn).set_index(['obs_time', 'picid'])
        except FileNotFoundError:
            continue

        # If a filter string was provided. Note(wtgee): Improve this.
        if table_filter:
            ref_table = full_ref_table.query(table_filter)
        else:
            ref_table = full_ref_table

        # Load the data and remove the bias.
        ref_psc = np.array(ref_table) - camera_bias
        psc_collection.append(ref_psc)

    # Big collection of PSCs.
    psc_collection = np.array(psc_collection)[:num_refs]
                                  
    # Get image times
    image_times = pd.to_datetime(ref_table.index.levels[0].values)

    # Slice frames from stamp collection - NOTE: could be combined with table_filter logic
    if frame_slice:
        psc_collection = psc_collection[:, frame_slice, :]
        image_times = image_times[frame_slice]

    # Get target PSC (may have changed with frame_slice)
    target_psc = psc_collection[0]
    num_frames = psc_collection.shape[1]

    # Get a normalized version of the entire stamp collection
    normalized_collection = np.array([pipeline.normalize(s) for s in psc_collection])

    # ### Build coeffecients
    # Get the coefficients the most optimally combine the normalized referenes into 
    # one single master reference.
    coeffs = pipeline.get_ideal_full_coeffs(normalized_collection)

    # Plot coefficients
    if make_plots:
        try:
            plot_coefficients(picid, coeffs[0], output_dir)
        except Exception as e:
            logger.error(f"Can't make coefficients plot for {picid}")
            logger.error(e)

    # ### Build reference PSC
    # Use the coeffecients generated from the normalized references and
    # apply them to the non-normalized (i.e. flux) stamps
    ideal_psc = pipeline.get_ideal_full_psc(
        psc_collection, 
        coeffs[0]
    ).reshape(num_frames, -1).astype(np.uint)

    if make_plots:
        try:
            plot_comparisons(picid, target_psc, ideal_psc, output_dir)
        except Exception as e:
            logger.error(f"Can't make stamp comparison plots for {picid}")
            logger.error(e)

    # ### Aperture photometry
    lc0 = pipeline.get_aperture_sums(
        target_psc, 
        ideal_psc, 
        image_times,
        aperture_size=aperture_size, 
        plot_apertures=make_plots,
        aperture_plot_path=os.path.join(output_dir, 'plots', 'apertures')
    )

    # Save the lightcurve dataframe to a csv file
    # NOTE: We do this before normalizing
    lc0.to_csv(os.path.join(output_dir, f'raw-flux-{picid}.csv'))

    if make_plots:
        try:
            plot_raw_lightcurve(picid, lc0, output_dir)
        except Exception as e:
            logger.error(f"Can't make raw lightcurve for {picid}")
            logger.error(e)
                                  

def plot_similarity(picid, similar_list, output_dir, num_stars=200):
    """ Plot of how the stars rank according to similarity. """
    fig = Figure()
    FigureCanvas(fig)
                                  
    ax = fig.add_subplot(111)
    ax.plot(similar_list.iloc[:num_stars].score)
                                  
    similar_fn = os.path.join(output_dir, 'plots', f'similar-source-ranks-{picid}.png')
    fig.savefig(similar_fn, transparent=False)

    
def plot_coefficients(picid, coeffs, output_dir):
    fig = Figure()
    FigureCanvas(fig)
    
    fig.set_size_inches(9, 6)
    ax = fig.add_subplot(111)
    
    ax.plot(coeffs)
    ax.set_xlabel('Reference Index')
    ax.set_ylabel('Coefficient Value')
    
    fig.suptitle(f'Reference coeffecients - {picid}')
    
    coeff_fn = os.path.join(output_dir, 'plots', f'coefficients-{picid}.png')
    fig.savefig(coeff_fn, transparent=False)
    
def plot_comparisons(picid, target_psc, ideal_psc, output_dir):
    num_frames = target_psc.shape[0]
    stamp_side = int(np.sqrt(target_psc.shape[1]))
    
    # Reshape into square stamps and plot the requested frame
    for frame_idx in range(num_frames):
        stamp_fig = plot.show_stamps([
            target_psc.reshape(num_frames, stamp_side, stamp_side), 
            ideal_psc.reshape(num_frames, stamp_side, stamp_side)
        ], frame_idx=frame_idx, show_residual=True, stretch='linear')

        stamp_fig.set_size_inches(9, 3.5)
        stamp_fig.suptitle(f'Target - Ref Comparison - {picid} - Frame: {frame_idx:03d}', 
                           y=0.98, fontsize=14)

        # Mask the brightest pixel
        t0 = target_psc[frame_idx].reshape(stamp_side, stamp_side)
        y_pos, x_pos = np.argwhere(t0 == t0.max())[0]
        center_color = helpers.pixel_color(x_pos, y_pos)
        stamp_fig.axes[0].scatter(x_pos, y_pos, marker='x', color='r')

        stamp_fig.tight_layout()
        target_ref_comp_fn = os.path.join(output_dir, 'plots', 'comparisons', f'ref-comparison-{picid}-{frame_idx:03d}.png')
        os.makedirs(os.path.dirname(target_ref_comp_fn), exist_ok=True)
        stamp_fig.savefig(target_ref_comp_fn, transparent=False)
                                  
def plot_raw_lightcurve(picid, lc0, output_dir):
    fig = Figure()
    FigureCanvas(fig)
    
    fig.set_size_inches(12, 7)
                                  
    ax = fig.add_subplot(111)
    
    ax.plot(lc0.loc[lc0.color == 'g'].target.values, marker='o', label='Target') #.plot(marker='o')
    ax.plot(lc0.loc[lc0.color == 'g'].reference.values, marker='o', label='Reference') #.plot(marker='o')
    
    fig.suptitle(f'Raw Flux - {picid}')
    fig.tight_layout()
    fig.legend()

    plot_fn = os.path.join(output_dir, 'plots', f'raw-flux-{picid}.png')
    fig.savefig(plot_fn, transparent=False)
                                  
def plot_normalized_lightcurve(picid, lc0, output_dir):
    plt.figure(figsize=(12, 6))
    i = 0
    for color in 'rgb':
        (lc1.loc[lc1.color == color].target + i).plot(marker='o', color=color, alpha=0.5)
        (lc1.loc[lc1.color == color].reference + i).plot(marker='x', color=color, alpha=0.5)
        i += .3

    plt.title(f'Normalized Flux per channel (+ offset) - {picid}')
    plt.legend()

    plot_fn = os.path.join(base_dir, f'normalized-flux-{picid}.png')
    plt.savefig(plot_fn)
                                  
    # Different
    plt.figure(figsize=(12, 6))
    i = 0
    for color in 'rgb':
    #     (lc1.loc[lc1.color == color].target + i).plot(marker='o', color=color, alpha=0.5)
    #     (lc1.loc[lc1.color == color].reference + i).plot(marker='x', color=color, alpha=0.5)
        t0 = lc1.loc[lc1.color == color].target
        r0 = lc1.loc[lc1.color == color].reference
        f0 = sigma_clip(t0 / r0, sigma=3)
        plt.plot((f0 + i), marker='o', ls='', alpha=0.5, color=color)
        i += .1

    # plt.ylim([.9, 1.1])

    plt.title(f'Normalized Flux per channel (+ offset) - {picid}')
    plt.legend()

    plot_fn = os.path.join(base_dir, f'normalized-flux-{picid}.png')
    plt.savefig(plot_fn)

def main(base_dir,
         camera_bias=2048,
         frame_slice=None,
         table_filter=None,
         num_refs=50,
         aperture_size=5,
         make_plots=False,
         picid=None,
         force=False,
         num_workers=8,
         chunk_size=12
    ):
                                  
    print(f'Building references for stars for observation in {base_dir}')

    if picid:
        print(f'Searching for picid={picid}')
        psc_files = glob(os.path.join(base_dir, 'stamps', f'{picid}.csv'))
    else:
        psc_files = glob(os.path.join(base_dir, 'stamps', '*.csv'))
        
    print(f'Found {len(psc_files)} PSC files')
    
    start_time = current_time()
    print(f'Starting at {start_time}')
                                  
    call_params = {
        'frame_slice': frame_slice,
        'table_filter': table_filter,
        'num_refs': num_refs,
        'camera_bias': camera_bias,
        'make_plots': make_plots,
        'aperture_size': aperture_size
    }
                                  
    # Build up the parameter list (NB: "clever" zip_longest usage)
    params = zip_longest(psc_files, [], fillvalue=call_params)

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
        picids = list(tqdm(
                executor.map(build_ref, params, chunksize=int(chunk_size)
            ), 
            total=len(psc_files))
        )
        print(f'Created {len(picids)} PSC references')
            
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
    parser.add_argument('--aperture-size', default=5, help="Aperture size for photometry")
    parser.add_argument('--picid', default=None, type=str, help="Create PSC only for given PICID")
    parser.add_argument('--make-plots', action='store_true', default=False, 
                        help="Create plots (increases time)")
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
        aperture_size=args.aperture_size,
        picid=args.picid,
        force=args.force,
        make_plots=args.make_plots,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size
    )
    print('Finished building reference PSC')
    
