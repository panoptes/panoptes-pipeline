import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from datetime import datetime
from datetime import timedelta

from pocs.utils.google.storage import PanStorage


def download_mlc(pic):
    """Download the master light curve from Google Cloud Storage"""
    pan_storage = PanStorage(bucket_name="panoptes-simulated-data")
    mlc_remote = "MLC/{}.json".format(pic)
    mlc_local = pan_storage.download(mlc_remote)
    with open(mlc_local, 'r') as mfile:
        mlc = json.load(mfile)
    return mlc


def make_plot(master, pic):
    """Make matplotlib plot of the master light curve for the given star.

    :param master: the master light curve in array format
    :param pic: the pic that the master light curve is for
    :return: the filename of the saved plot
    """
    # Unpack light curve into arrays
    mtimes = []
    r, g, b = [], [], []
    for j in range(len(master)):
        mt = master[j]
        mtime = datetime.strptime(mt['time'], '%Y-%m-%dT%H:%M:%S.%f')
        mtimes.append(mtime)
        r.append(mt['R'])
        g.append(mt['G'])
        b.append(mt['B'])

    # Plot R G and B vs. time
    fig, ax = plt.subplots()
    mtimes, r = (list(t) for t in zip(*sorted(zip(mtimes, r))))
    mtimes, g = (list(t) for t in zip(*sorted(zip(mtimes, g))))
    mtimes, b = (list(t) for t in zip(*sorted(zip(mtimes, b))))
    ax.plot(mtimes, b, color='blue', linewidth=0.2, label='B flux')
    ax.plot(mtimes, r, color='red', linewidth=0.2, label='R flux')
    ax.plot(mtimes, g, color='green', linewidth=0.2, label='G flux')

    # Format plot
    plt.ylim(0.85, 1.15)
    fig.autofmt_xdate()
    days = mdates.DayLocator()
    ax.xaxis.set_major_locator(days)
    day_fmt = mdates.DateFormatter('%m-%d-%Y')
    ax.xaxis.set_major_formatter(day_fmt)
    datemin = min(mtimes) - timedelta(0.2)
    datemax = max(mtimes) + timedelta(0.2)
    plt.xlim(datemin, datemax)
    plt.title('Master Light Curve for {}'.format(pic))
    plt.xlabel('Time of observation')
    plt.ylabel('Flux (normalized)')
    rline = mlines.Line2D([], [], color='red', linewidth=2)
    gline = mlines.Line2D([], [], color='green', linewidth=2)
    bline = mlines.Line2D([], [], color='blue', linewidth=2)
    plt.legend(handles=[rline, gline, bline], labels=['R flux', 'G flux', 'B flux'])

    # Save plot to local file
    filename = '{}/plots/MLC_{}.png'.format(os.getenv('PANDIR'), pic)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)

    return filename


def main(pic):
    """Make plot of master light curve"""
    master = download_mlc(pic)
    filename = make_plot(master, pic)
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pic", type=str, nargs='+', help="The PIC id of the star to plot a light curve for.")
    args = parser.parse_args()
    for pic in args.pic:
        main(pic)

