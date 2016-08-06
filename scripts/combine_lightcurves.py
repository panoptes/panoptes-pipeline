#!/usr/bin/env python3

import os
import json
import argparse
import warnings

from pocs.utils.google.storage import PanStorage


class LightCurveCombiner(object):

    """
    Class to combine light curve segments into master light curve
    """

    def __init__(self, bucket=None, storage=None):
        assert bucket is not None, warnings.warn("A valid bucket is required.")
        if storage is None:
            storage = PanStorage(bucket=bucket)
        self.storage = storage

    def run(self, pic):
        """Build a master light curve for a given PIC and output it as JSON.

        :param pic: The PIC object to combine light curves for
        """
        temp_dir = '/tmp/lc-combine'
        curves = self.get_curves_for_pic(pic, temp_dir)
        master = self.combine_curves(curves)
        self.upload_output(pic, master, temp_dir)

    def get_curves_for_pic(self, pic, temp_dir):
        """Read light curve files and add all curves to array.

        :param pic: the PIC id for the star to make a master light curve for
        :return: an array of all the light curves for the given PIC
        """
        curves = []
        prefix = "LC/{}/".format(pic)

        pan_storage = self.storage
        blobs = pan_storage.list_remote(prefix)

        num_blobs = 0
        for blob in blobs:
            local_path = "{}/{}".format(temp_dir, blob.name)
            path = pan_storage.download(blob.name, local_path=local_path)
            with open(path, 'r') as f:
                try:
                    curve = json.load(f)
                    curves.append(curve)
                except ValueError as err:
                    print("Error: Object {} could not be decoded as JSON: {}".format(
                        blob.name, err))
            num_blobs += 1
        if num_blobs == 0:
            raise NameError("No light curves with PIC '{}' found in bucket '{}'.".format(
                pic, pan_storage.bucket.name))
        return curves

    def combine_curves(self, curves):
        """Flatten all the data points in the curves into one master curve array.

        Each data point has a timestamp, exposure duration, RGB fluxes, and
        RGB flux uncertainties.
        :param curves: an array of light curves, each with many data points, to be combined
        :return: a master light curve stored in a single array
        """
        master = []
        for c in curves:
            for data_point in c:
                master.append(data_point)
        return master

    def upload_output(self, pic, data, temp_dir):
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

        pan_storage = self.storage
        pan_storage.upload(local_path, remote_path=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pic', type=str, help="The PIC ID of the star to build a"
                                              " master light curve for.")
    args = parser.parse_args()
    combiner = LightCurveCombiner(bucket='panoptes-simulated-data')
    combiner.run(args.pic)
