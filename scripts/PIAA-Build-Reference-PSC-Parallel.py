#!/usr/bin/env python3

# ## Build Reference PSC

import os
import numpy as np
import pandas as pd
from copy import copy

from glob import glob

from matplotlib import pyplot as plt
plt.style.use('bmh')
from IPython import display

from astropy.time import Time
from astropy.stats import sigma_clip

from piaa.exoplanets import TransitInfo, get_exoplanet_transit
from piaa.utils import helpers
from piaa.utils import plot
from piaa.utils import pipeline


def main(base_dir,
         camera_bias=2048,
         frame_slice=None,
         table_filter=None,
         num_refs=50
    ):
    
    # ### Get stamp collection
    # Build up a stamp book of the target star and the top references.
    stamp_files = glob(os.path.join(base_dir, 'stamps', '*.csv'))
    
    # For picid in stamp_files
    
def build_ref(picid,
              base_dir,
              aperture_size=5,
              plot_apertures=False
    ):
    
    # Subdirectory to place processed files
    processed_dir = os.path.normpath(os.path.join(base_dir, 'processed', str(picid))
    
    # Get the list of similar stars. The picid at index 0 is the target star
    collection_fn = os.path.join(base_dir, 'stamps', 'similar', f'{picid}.csv')
    similar_stars = pd.read_csv(collection_fn, names=['picid', 'v'])

    # Plot of Similarity
    #plt.plot(similar_stars.iloc[:stamp_book_size].v)

    ref_list = list(similar_stars.picid[:stamp_book_size].values)

    # Build stamp collection
    stamp_collection = list()
    for i, ref_picid in enumerate(ref_list):
        stamp_fn = os.path.join(base_dir, 'stamps', f'{ref_picid}.csv')

        try:
            full_ref_table = pd.read_csv(stamp_fn).set_index(['obs_time', 'picid'])
        except FileNotFoundError:
            continue

        # If a filter string was provided
        if table_filter:
            ref_table = full_ref_table.query(table_filter)
        else:
            ref_table = full_ref_table

        ref_psc = np.array(ref_table) - camera_bias
        stamp_collection.append(ref_psc)

    stamp_collection = np.array(stamp_collection)

    # Slice frames from stamp collection - NOTE: could be combined with table_filter logic
    stamp_collection = stamp_collection[:, frame_slice, :]
    
    # Get image times
    image_times = pd.to_datetime(ref_table.index.levels[0].values)[frame_slice]

    # Get target PSC (may have changed with frame_slice)
    target_psc = stamp_collection[0]
    num_frames = stamp_collection.shape[1]

    # Get a normalized version of the entire stamp collection
    normalized_collection = np.array([pipeline.normalize(s) for s in stamp_collection[:num_refs]])

    # ### Build coeffecients
    # Get the coefficients the most optimally combine the normalized referenes into 
    # one single master reference.
    coeffs = pipeline.get_ideal_full_coeffs(normalized_collection)

    # Plot coefficients
    plot_coefficients(picid, coeffs[0], processed_dir)

    # ### Build reference PSC
    # Use the coeffecients generated from the normalized references and
    # apply them to the non-normalized (i.e. flux) stamps
    ideal_psc = pipeline.get_ideal_full_psc(
        stamp_collection[:num_refs], 
        coeffs[0]
    ).reshape(num_frames, -1)

    plot_comparisons(picid, target_psc, ideal_psc, processed_dir):

    # ### Aperture photometry
    csv_file = os.path.join(processed_dir, '{}_{}_lc.csv'.format(
        sequence.replace('/', '_'), 
        picid
    ))
    
    # Do the actual aperture photometry
    lc0 = pipeline.get_aperture_sums(
        target_psc, 
        ideal_psc, 
        image_times,
        aperture_size=aperture_size, 
        plot_apertures=plot_apertures,
        aperture_fn=os.path.join(base_dir, 'apertures', f'{picid}')
    )

    # Save the lightcurve dataframe to a csv file
    # NOTE: We do this before normalizing
    # Save before adding relative flux
    if save_fn:
        lc0.to_csv(save_fn)

    plt.figure(figsize=(12, 6))
    plt.plot(lc0.loc[lc0.color == 'g'].target.values, marker='o', label='Target') #.plot(marker='o')
    plt.plot(lc0.loc[lc0.color == 'g'].reference.values, marker='o', label='Reference') #.plot(marker='o')
    plt.title(f'Raw Flux - {picid} - {sequence}')
    plt.tight_layout()
    plt.legend()

    plot_fn = os.path.join(base_dir, f'raw-flux-{picid}-{sequence}.png')
    plt.savefig(plot_fn)


    # In[28]:


    # Make a copy
    lc1 = lc0.copy()


    # In[29]:


    for field in ['reference', 'target']:
        color_normer = lc0[:30].groupby('color')[field].apply(lambda x: np.mean(x))

        for color, normalizer in color_normer.iteritems():
            print(f"{field} {color} μ={normalizer:.04f}")

            # Get the raw values
            raw_values = lc1.loc[lc1.color == color, (field)]

            lc1.loc[lc1.color == color, (f'{field}')] = (raw_values / normalizer)


    # In[30]:


    plt.figure(figsize=(12, 6))
    i = 0
    for color in 'rgb':
        (lc1.loc[lc1.color == color].target + i).plot(marker='o', color=color, alpha=0.5)
        (lc1.loc[lc1.color == color].reference + i).plot(marker='x', color=color, alpha=0.5)
        i += .3

    plt.title(f'Normalized Flux per channel (+ offset) - {picid} - {sequence}')
    plt.legend()

    plot_fn = os.path.join(base_dir, f'normalized-flux-{picid}-{sequence}.png')
    plt.savefig(plot_fn)


    # In[31]:


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

    plt.title(f'Normalized Flux per channel (+ offset) - {picid} - {sequence}')
    plt.legend()

    plot_fn = os.path.join(base_dir, f'normalized-flux-{picid}-{sequence}.png')
    plt.savefig(plot_fn)


    # ### Exoplanet info
    # 
    # See bottom of notebook

    # In[32]:


    transit_times = TransitInfo(
        Time('2018-08-22 04:53:00'),
        Time('2018-08-22 05:47:00'),
        Time('2018-08-22 06:41:00')
    )

    transit_datetimes = [transit_times.midpoint.datetime, transit_times.ingress.datetime, transit_times.egress.datetime]


    # In[33]:


    for color, data in lc1.groupby('color'):

        y = pd.Series(sigma_clip(data.target / data.reference), index=lc1.index.unique()).dropna()
        x = y.index

    #     base_model_flux = np.ones_like(y)
        base_model_flux = get_exoplanet_transit(x, transit_times=transit_times)

        lc_fig = plot.plot_lightcurve(x, y, base_model_flux, 
            transit_info=transit_datetimes,
            title="{} Lightcurve - {}".format(sequence.replace('_', ' '), color),
            ylim=[.9, 1.1], 
            color=color
        )    
        lc_fig.get_axes()[0].set_ylim([0.9, 1.1])

        print('σ={:.04%}'.format(y.std()))

        display.display(lc_fig)


    # ### Examine

    # In[34]:


    # Detrend by the median value
    t0 = target_psc.sum(1) / np.mean(target_psc.sum(1))
    i0 = ideal_psc.sum(1) / np.mean(ideal_psc.sum(1))


    # In[35]:


    plt.figure(figsize=(12, 9))
    plt.plot(target_psc.sum(1), marker='o', c='r', label='Target')
    plt.plot(ideal_psc.sum(1), marker='o', c='b', label='Reference')

    # plt.axvline(midpoint, label='mid-transit', ls='-.', c='g')
    # plt.axvline(ingress, label='ingress', ls='--')
    # plt.axvline(egress, label='egress', ls='--')

    plt.title(f'Target vs Reference Full Stamp Sum {picid}')
    plt.legend()

    
def plot_coefficients(picid, coeffs, save_dir):
    fig = Figure()
    FigureCanvas(fig)
    
    fig.set_size_inches(9, 6)
    ax = fig.add_subplot(111)
    
    ax.plot(coeffs)
    ax.set_xlabel('Reference Index')
    ax.set_ylabel('Coefficient Value')
    
    fig.suptitle(f'Reference coeffecients - {picid}')
    
    coeff_fn = os.path.join(save_dir, f'ref-comparison-{picid}.png')
    fig.savefig(coeff_fn, transparent=False)
    
def plot_comparisons(picid, target_psc, ideal_psc, frame_idx, save_dir):
    stamp_side = int(np.sqrt(target_psc.shape[1]))
    
    # Reshape into square stamps and plot the requested frame
    stamp_fig = plot.show_stamps([
        target_psc.reshape(num_frames, stamp_side, stamp_side), 
        ideal_psc.reshape(num_frames, stamp_side, stamp_side)
    ], frame_idx=frame_idx, show_residual=True, stretch='linear')

    stamp_fig.set_size_inches(9, 3.1)
    stamp_fig.suptitle(f'Target - Ref Comparison - {picid} - Frame: {frame_idx:03d}', 
                       y=1.0, fontsize=14)

    # Mask the brightest pixel
    t0 = target_psc[frame_idx].reshape(stamp_side, stamp_side)
    y_pos, x_pos = np.argwhere(t0 == t0.max())[0]
    center_color = helpers.pixel_color(x_pos, y_pos)
    stamp_fig.axes[0].scatter(x_pos, y_pos, marker='x', color='r')

    stamp_fig.tight_layout()
    target_ref_comp_fn = os.path.join(save_dir, f'ref-comparison-{picid}-{frame_idx:03d}.png')
    stamp_fig.savefig(target_ref_comp_fn, transparent=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Find similar stars for each star.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--directory', default=None, type=str,
                       help="Directory containing observation images.")
    parser.add_argument('--picid', default=None, type=str, help="Create PSC only for given PICID")
    parser.add_argument('--stamp-book-size', default=200, help="How many PSCs to put in stamp book")
    parser.add_argument('--num-workers', default=8, help="Number of workers to use")
    parser.add_argument('--chunk-size', default=10, help="Chunks per worker")
    parser.add_argument('--force', action='store_true', default=False, 
                        help="Force creation (deletes existing files)")

    args = parser.parse_args()

    fields_dir = os.path.join(os.environ['PANDIR'], 'images', 'fields')
    base_dir = os.path.join(fields_dir, args.directory)
    assert os.path.isdir(base_dir)
    
    unit_id, field_id, cam_id, seq_id = args.directory.split('/')
    sequence = '_'.join([unit_id, cam_id, seq_id])

    print(f'Using {args.num_workers} workers with {args.chunk_size} chunks')
    main(
        base_dir=args.directory,
        picid=args.picid,
        force=args.force,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size
    )
    print('Finished building reference PSC')
    