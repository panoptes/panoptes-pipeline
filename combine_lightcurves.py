import os
import json
import argparse

import glob


def get_curves_for_pic(pic):
    curves = []
    for filename in glob.iglob("data/LC/{}/**/*.json".format(pic), recursive=True):
        with open(filename, 'r') as f:
            c = json.load(f)
            curves.append(c)
    return curves


def combine_curves(curves):
    master = []
    for c in curves:
        for t in c:
            master.append(t)
    return master


def write_output(output, pic):
    filename = 'output/{}.json'.format(pic)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fo:
        json.dump(output, fo)


def main(pic):
    curves = get_curves_for_pic(pic)
    master = combine_curves(curves)
    write_output(master, pic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pic', type=str, help="The PICID of the star to build a light curve for.")
    args = parser.parse_args()
    main(args.pic)

