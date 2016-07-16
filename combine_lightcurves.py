import json
import argparse


def read_input(filenames):
    curves = []
    for fn in filenames:
        with open(fn, 'r') as f:
            c = json.load(f)
            curves.append(c)
    return curves


def combine_curves(curves):
    master = []
    for c in curves:
        for t in c:
            master.append(t)
    return master


def write_output(output, name):
    with open(name, 'w') as fo:
        json.dump(output, fo)


def main(filenames):
    curves = read_input(filenames)
    master = combine_curves(curves)
    write_output(master, 'LC_0001_0003.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('PSC1', type=string, nargs='+')
    parser.add_argument('PSC2', type=string, nargs='+')
    args = parser.parse_args()
    main([args.PSC1, args.PSC2])