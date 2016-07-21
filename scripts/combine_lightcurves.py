import os
import json
import argparse

import glob
from gcloud import storage


def get_curves_for_pic(pic):
    """Read light curve files and add all curves to array.

    :param pic: the PIC id for the star to make a master light curve for
    :return: an array of all the light curves for the given PIC
    """
    curves = []
    for filename in glob.iglob("LC/{}/**/*.json".format(pic), recursive=True):
        with open(filename, 'r') as f:
            c = json.load(f)
            curves.append(c)
    return curves


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
        for datapoint in c:
            master.append(datapoint)
    return master


def write_output(filename, data):
    """Write the data to a local file.

    :param filename: the filename for the output
    :param data: the master light curve
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fo:
        json.dump(data, fo)


def upload_to_cloud(filename, data):
    """Upload the data to a Google Cloud Storage bucket.

    :param filename: the filename for the output
    :param data: the master light curve
    """
    projectid = 'panoptes-survey'
    client = storage.Client(project=projectid)
    bucket_name = 'panoptes-simulated-data'

    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_string(data)


def build_mlc(pic, tocloud):
    """Build a master light curve for a given PIC and output it as JSON.

    :param pic: The PIC object to combine light curves for
    :param tocloud: if True, data will also be upload to Google Cloud Storage
    """
    curves = get_curves_for_pic(pic)
    master = combine_curves(curves)
    filename = 'MLC/{}.json'.format(pic)
    write_output(filename, master)
    if tocloud:
        master_json = json.dump(master)
        upload_to_cloud(filename, master_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pic', type=str, help="The PICID of the star to build a"
                                              " light curve for.")
    parser.add_argument('-c', action='store_true', help='Store the output in a'
                        ' Cloud Storage bucket in addition to writing a local copy.')
    args = parser.parse_args()
    build_mlc(args.pic, args.c)
