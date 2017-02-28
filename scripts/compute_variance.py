#!/usr/bin/env python

from multiprocessing import Pool

import numpy as np
import pandas as pd

from astropy import units as u
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

    print("Getting variance for index {}".format(args.target_index))
    print(obs.point_sources.iloc[args.target_index])

    start = Time.now()
    print("Starting at  {}".format(start))

    # obs.get_variance_for_target(args.target_index, normalize=True)

    num_sources = len(obs.point_sources)

    v = np.zeros((num_sources), dtype=np.float)

    s0 = obs.get_source_stamps(args.target_index)
    stamp0 = np.array([s.data.flatten() for s in s0])
    stamp0 = stamp0 / stamp0.sum()

    def get_v(source_index):
        s1 = obs.get_source_stamps(source_index)
        stamp1 = np.array([s.data.flatten() for s in s1])
        stamp1 = stamp1 / stamp1.sum()

        return ((stamp0 - stamp1)**2).sum()

    with Pool() as pool:
        result = pool.map(get_v, obs.point_sources.index)

    obs.point_sources['V'] = pd.Series(result)

    # Sort the values by lowest total variance
    obs.point_sources.sort_values(by=['V'], inplace=True)

    # Save values to file
    obs.point_sources.to_csv('vary_{}.csv'.format(args.target_index))

    print("Processing time: {:02f}".format(((Time.now() - start).sec * u.second)))
