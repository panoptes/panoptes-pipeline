#!/usr/bin/env python3

import os
import h5py                                   
from glob import glob
from tqdm import tqdm
from contextlib import suppress

import numpy as np

from astropy.stats import sigma_clip

from piaa.utils import pipeline
from piaa.utils import plot
from pocs.utils import current_time

import logging

BIAS=2048

def main(
        stamp_file, 
        aperture_size=6, 
        num_references=101, 
        show_progress=True, 
        make_plots=False,
        *args, **kwargs):
    try:
        stamps = h5py.File(stamp_file)
    except FileNotFoundError:
        logging.warning("File not found: {}".format(stamp_file))
        return
    
    star_iterator = enumerate(list(stamps.keys()))
    
    base_dir = os.path.dirname(stamp_file)
    sequence = os.path.splitext(os.path.basename(stamp_file))[0]
    
    process_dir = os.path.join(base_dir, 'processed')
    
    os.makedirs(process_dir, exist_ok=True)
    
    # Show progress bar
    if show_progress:
        star_iterator = tqdm(star_iterator, total=len(list(stamps.keys())), desc='Looping sources')

    for i, picid in star_iterator:
        if picid not in stamps:
            logging.debug("Skipping {} - no stamp".format(picid))
            continue
            
        if 'similar_stars' not in stamps[picid]:
            logging.debug("Skipping {} - no similar sources".format(picid))
            continue

        vary_series = np.array(stamps[picid]['similar_stars'])
        
        # Get the target and the top-matching references
        stamp_collection = np.array([
            pipeline.get_psc(str(idx), stamps) - BIAS 
            for idx in vary_series[:num_references]])
        
        # Get target PSC
        target_psc = stamp_collection[0]
        
        num_frames = target_psc.shape[0]
        stamp_side = int(np.sqrt(target_psc.shape[1]))
        
        # Get a normalized version of the entire stamp collection
        normalized_collection = np.array([pipeline.normalize(s) for s in stamp_collection])
        
        # Get the coefficients the most optimally combine the normalized referenes into 
        # one single master reference.
        coeffs = pipeline.get_ideal_full_coeffs(normalized_collection)
        
        stamps[picid]['coeffs'] = coeffs[0]
        stamps[picid]['coeffs'].attrs['date'] = current_time(flatten=True)
        
        # Use the coeffecients generated from the normalized references and
        # apply them to the non-normalized (i.e. flux) stamps
        ideal_psc = pipeline.get_ideal_full_psc(
            stamp_collection, 
            coeffs[0]
        ).reshape(num_frames, -1)
        
        stamps[picid]['reference'] = ideal_psc
        
        image_times = np.array(stamps.attrs['image_times'])
        
        lc0 = pipeline.get_aperture_sums(
            target_psc, 
            ideal_psc, 
            image_times, 
            aperture_size=aperture_size, 
        )
        
        # No normalization
        lc0['rel_flux'] = sigma_clip(lc0.target / lc0.reference).filled(99)
        lc0 = lc0.loc[lc0.rel_flux != 99]
        
        # Get the mean for each channel
        color_means = lc0.groupby('color').rel_flux.mean()

        # Make a copy
        lc2 = lc0.copy()

        # Normalize by mean of color channel
        for color, mean in color_means.iteritems():
            logging.debug("{} μ={:.04f}".format(color, mean))

            # Get the raw values
            raw_values = lc2.loc[lc2.color == color, ('rel_flux')]

            mean_values = sigma_clip(raw_values / mean).filled(99)

            lc2.loc[lc2.color == color, ('rel_flux')] = mean_values

        # Get the light curves
        for color, data in lc2.groupby('color'):
            y = data.loc[data.rel_flux != 99].rel_flux
            x = y.index
            
            stamps[picid]['reference'].attrs['rms_{}'.format(color)] = y.std()

            if make_plots:
                base_model_flux = np.ones_like(y)

                lc_fig = plot.plot_lightcurve(x, y, base_model_flux, 
                    title="{} Lightcurve - {}".format(sequence.replace('_', ' '), color),
                    ylim=[.9, 1.1], 
                    color=color
                )    
                lc_fig.suptitle('PICID {}'.format(picid), y=0.95, fontsize=16)    

                logging.info('{} σ={:.04%}'.format(color, y.std()))
            
                color_channel_fn = os.path.join(
                    process_dir,
                    'lc_{}_{}.png'.format(picid, color)
                )

                lc_fig.savefig(color_channel_fn, dpi=150, transparent=False)
                
        # Combined color channels
        lc1 = lc0.groupby('obstime').sum()

        # Redo the relative flux with new sums
        lc1['rel_flux'] = sigma_clip(lc1.target / lc1.reference).filled(99)
        lc1 = lc1.loc[lc1.rel_flux != 99]

        # Normalize by the mean
        lc1['rel_flux'] = sigma_clip(lc1.rel_flux / lc1.rel_flux.mean()).filled(99)
        lc1 = lc1.loc[lc1.rel_flux != 99]
        
        stamps[picid]['reference'].attrs['rms'] = lc1.rel_flux.std()
        
        # Plot the lightcurve
        x = lc1.index
        y = lc1.rel_flux
        
        if make_plots:
            base_model_flux = np.ones_like(lc1.rel_flux.values)
            
            lc_fn = os.path.join(
                process_dir,
                'lc_{}.png'.format(picid, color)
            )

            lc_fig = plot.plot_lightcurve(x, y, base_model_flux, 
                title="{} Lightcurve".format(sequence.replace('_', ' ')),
                ylim=[.9, 1.1]
            )
            lc_fig.suptitle('PICID {}'.format(picid), y=0.95, fontsize=16)    

            lc_fig.savefig(lc_fn, dpi=150, transparent=False)
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create reference PSC for each target")
    parser.add_argument('--stamp_file', required=True, type=str, help="HDF5 Stamps file")
    parser.add_argument('--aperture_size', default=6, help="Final photometry aperture")
    parser.add_argument('--num_references', default=101, help="Number of references to use to build ideal refrence.")
    parser.add_argument('--log_level', default='debug', help="Log level")
    parser.add_argument('--log_file', help="Log files, default $PANLOG/create_stamps_<datestamp>.log")
    parser.add_argument('--show_progress', default=True, action='store_true',  help="Show progress bars")
    parser.add_argument('--make_plots', default=False, action='store_true',  help="Make plots")
    
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
        print("References created {}".format(stamps_fn))
