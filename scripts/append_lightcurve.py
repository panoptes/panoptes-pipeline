import argparse
import json

from pocs.utils.google.storage import PanStorage


#############################################################################
# This file is not currently in use, but would be used if we switched       #
# to a different storage solution that allowed for appending to the master  #
# light curve rather than recombining from source.                          #
#############################################################################

def download_lightcurve(lc_filename, storage):
    """Download the light curve (LC) from Google Cloud Storage."""
    lc_json = storage.download_string(lc_filename)
    lc = json.loads(lc_json.decode())
    return lc


def get_pic_from_filename(lc_filename):
    """Extract the Panoptes Input Catalog (PIC) name from the LC filename."""
    for dirname in lc_filename.split('/'):
        if dirname.startswith("PIC"):
            pic = dirname
            return pic
    raise Exception("Error: PIC not detected from {}.".format(lc_filename))


def append_to_mlc(lc, pic, storage):
    """Download the master light curve, add the new LC and reupload."""
    mlc_filename = "MLC/{}.json".format(pic)
    if mlc_filename in storage.list_remote():
        mlc_json = storage.download_string(mlc_filename)
        mlc = json.loads(mlc_json)
    else:
        mlc = []
    for entry in lc:
        mlc.append(entry)
    storage.upload_string(json.dumps(mlc), mlc_filename)
    print("SUCCESS! {} updated.".format(mlc_filename))


def main(lc_filename, storage):
    """Append the given light curve to the existing master light curve.

    :param lc_filename: the name of the light curve file to append
    :param storage: the storage object to use to download and upload data
    """
    lc = download_lightcurve(lc_filename, storage)
    pic = get_pic_from_filename(lc_filename)
    append_to_mlc(lc, pic, storage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lc_filename", type=str,
                        help="The filename of the light curve to append to the master light curve for that star.")
    storage = PanStorage(project_id="panoptes-survey",
                         bucket_name="panoptes-simulated-data")
    args = parser.parse_args()
    main(args.lc_filename, storage)
