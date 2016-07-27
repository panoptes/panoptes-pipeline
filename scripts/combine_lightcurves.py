#!/usr/bin/env python3

import os
import shutil
import json
import argparse
from json.decoder import JSONDecodeError

from pocs.utils.google.storage import PanStorage


def get_curves_for_pic(pic):
    """Read light curve files and add all curves to array.

    :param pic: the PIC id for the star to make a master light curve for
    :return: an array of all the light curves for the given PIC
    """
    curves = []
    bucket_name = 'panoptes-simulated-data'
    prefix = "LC/{}/".format(pic)
    temp_dir = '{}/temp'.format(os.getenv('PANDIR', default='/var/panoptes'))

    pan_storage = PanStorage(bucket=bucket_name)
    blobs = pan_storage.list_remote(prefix)

    for blob in blobs:
        local_path = "{}/{}".format(temp_dir, blob.name)
        path = pan_storage.download(blob.name, local_path=local_path)
        with open(path, 'r') as f:
            try:
                curve = json.load(f)
                curves.append(curve)
            except JSONDecodeError:
                print("Error: Object could not be decoded as JSON.")
    return curves, temp_dir


def combine_curves(curves):
    """Flatten all the data points in the curves into one master curve array.

    Each data point has a timestamp, exposure duration, RGB fluxes, and
    RGB flux uncertainties. Eventually should add functionality for
    overlapping exposures and be much less naive.
    :param curves: an array of light curves to be combined
    :return: a master light curve stored in a single array
    """
    master = []
    for c in curves:
        for data_point in c:
            master.append(data_point)
    return master


def upload_output(pic, data, temp_dir):
    """Write the data to a local file.

    :param pic: the PIC id of the file to uploaded
    :param data: the master light curve
    :param temp_dir: the path to the temporary directory where the data is stored
    """
    filename = 'MLC/{}.json'.format(pic)
    local_path = '{}/{}'.format(temp_dir, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'w') as fo:
        json.dump(data, fo)

    bucket_name = 'panoptes-simulated-data'
    pan_storage = PanStorage(bucket=bucket_name)
    pan_storage.upload(local_path, remote_path=filename)


def cleanup(temp_dir):
    """Remove the temporary directory and its contents."""
    shutil.rmtree(temp_dir)


def build_mlc(pic):
    """Build a master light curve for a given PIC and output it as JSON.

    :param pic: The PIC object to combine light curves for
    """
    curves, temp_dir = get_curves_for_pic(pic)
    master = combine_curves(curves)
    upload_output(pic, master, temp_dir)
    cleanup(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pic', type=str, help="The PICID of the star to build a"
                                              " master light curve for.")
    args = parser.parse_args()
    build_mlc(args.pic)
