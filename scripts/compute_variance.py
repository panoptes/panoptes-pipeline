#!/usr/bin/env python

from astropy.time import Time
from astropy.utils.console import ProgressBar

from piaa.observation import Observation


def add_result(result):
    print(result.get())


def show_error(e):
    print(e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True, type=str, help="Image directory containing FITS files")
    parser.add_argument('--target-index', type=int, default=None, help="Target index to compute variance for")
    parser.add_argument('--all', action="store_true", default=False, help="Get variance for all targets")
    parser.add_argument('--subtract', action="store_true", default=False, help="Subtract background from data cube")
    parser.add_argument('--create-stamps', action="store_true", default=False,
                        help="Create the stamps for each source")
    parser.add_argument('--log-level', default='INFO', help="Log level: INFO or DEBUG")
    parser.add_argument('--verbose', action="store_true", default=False, help="Show output")
    args = parser.parse_args()

    obs = Observation(args.image_dir, log_level=args.log_level)

    start = Time.now()
    print("Starting at  {}".format(start))

    # Normalize first
    if args.subtract:
        print("Creating background estimates")
        obs.subtract_background()
        subtracting_done = Time.now()
        print("Subtracting done: {:02f} seconds".format(((subtracting_done - start).sec)))

    if args.create_stamps:
        print("Creating stamps for point sources")
        obs.create_stamps(ipython_widget=args.verbose)
        stamps_done = Time.now()
        print("Stamp creation done: {:02f} seconds".format(((stamps_done - start).sec)))

    if args.target_index is not None:
        print("Getting variance for index {}".format(args.target_index))
        print(obs.point_sources.iloc[args.target_index])

        obs.get_variance_for_target(args.target_index)
        print("Variance done: {:02f} seconds".format(((Time.now() - stamps_done).sec)))
    elif args.all:
        print("Getting variance for all sources")
        for source_index in ProgressBar(obs.point_sources.index):
            obs.get_variance_for_target(source_index, show_progress=False)

    print("Total time: {:02f} seconds".format(((Time.now() - start).sec)))
