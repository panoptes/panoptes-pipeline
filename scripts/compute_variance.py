#!/usr/bin/env python

import pandas as pd

from astropy.time import Time

from piaa.observation import Observation


def add_result(result):
    print(result.get())


def show_error(e):
    print(e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, help="Image directory containing FITS files")
    parser.add_argument('--target-index', type=int, help="Target index to compute variance for")
    args = parser.parse_args()

    obs = Observation(args.image_dir)

    start = Time.now()
    print("Starting at  {}".format(start))

    # Normalize first
    print("Normalizing stamps")
    obs.create_normalized_stamps()
    print("Normalization done: {:02f} seconds".format(((Time.now() - start).sec)))

    print("Getting variance for index {}".format(args.target_index))
    print(obs.point_sources.iloc[args.target_index])
    result = obs.get_variance_for_target(args.target_index, ipython_widget=True)

    obs.point_sources['V'] = pd.Series(result)

    # Sort the values by lowest total variance
    obs.point_sources.sort_values(by=['V'], inplace=True)

    # Save values to file
    output_file = 'vary_{}.csv'.format(args.target_index)
    print("Saving to file: {}".format(output_file))
    obs.point_sources.to_csv(output_file)

    print("Processing time: {:02f} seconds".format(((Time.now() - start).sec)))
