import argparse
import json

from pocs.utils.google.storage import PanStorage


def main(lc_filename, storage):

    lc_json = storage.download_string(lc_filename)
    print(lc_json.decode())
    lc = json.loads(lc_json.decode())

    for dirname in lc_filename.split('/'):
        if dirname.startswith("PIC"):
            pic = dirname
    if pic is None:
        raise Exception("Error: PIC not detected from {}.".format(lc_filename))

    mlc_filename = "MLC/{}.json".format(pic)
    print(storage.list_remote())
    if mlc_filename in storage.list_remote():
        mlc_json = storage.download_string(mlc_filename)
        mlc = json.loads(mlc_json)
    else:
        mlc = []

    for entry in lc:
        mlc.append(entry)

    print(mlc)
    storage.upload_string(json.dumps(mlc), mlc_filename)
    print("after")
    print(storage.list_remote())

    print("SUCCESS! {} updated.".format(mlc_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lc_filename", type=str, help="The filename of the light curve to append to the master light curve for that star.")
    storage = PanStorage(project_id="panoptes-survey", bucket_name="panoptes-simulated-data")
    args = parser.parse_args()
    main(args.lc_filename, storage)
