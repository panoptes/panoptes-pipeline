#!/usr/bin/env python

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
    obs.create_subtracted_stamps()
    normal_done = Time.now()
    print("Normalization done: {:02f} seconds".format(((normal_done - start).sec)))

    print("Getting variance for index {}".format(args.target_index))
    print(obs.point_sources.iloc[args.target_index])

    obs.get_variance_for_target(args.target_index)
    print("Variance done: {:02f} seconds".format(((Time.now() - normal_done).sec)))

    print("Total time: {:02f} seconds".format(((Time.now() - start).sec)))
