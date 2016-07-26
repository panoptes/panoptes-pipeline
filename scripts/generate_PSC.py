import argparse
import astropy.units as u
import json
import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy.io import fits
from datetime import datetime as dt
from datetime import timedelta as tdelta


def main(**kwargs):

    verbose = kwargs.get('verbose', False)
    output_dir = kwargs.get('output_dir')

    # Generate Fake Postange Stamp Cube (FITS cube)
    sky_background = 1000.
    sky_sigma = 5.
    nx = 12
    ny = 16
    nt = 42
    data_cube = np.random.normal(sky_background, sky_sigma, (nt, ny, nx))
    obstime = dt.utcnow()
    unit = 'PAN000'  # I've given this the ID of PAN000 just in case it gets confused for data from a real unit.
    camera = '0x2A'
    target_name = 'faketarg'
    seq_id = '{}{}_{}'.format(unit, camera, obstime.strftime('%Y%m%d_%H%M%SUT'))
    xpixorg = 1042
    ypixorg = 2042
    exptime = 100.  # seconds
    c = SkyCoord.from_name('HR8799', frame='fk5')

    hdu = fits.PrimaryHDU(data_cube)
    metadata = {'SEQID': seq_id,
                'FIELD': target_name,
                'RA': c.ra.to(u.degree).value,
                'DEC': c.dec.to(u.degree).value,
                'EQUINOX': c.equinox.value,
                'OBSTIME': obstime.isoformat(),
                'XPIXORG': xpixorg,
                'YPIXORG': ypixorg,
                }

    for t in range(nt):
        # slightly randomize time gap between images
        gap = tdelta(0, exptime + np.random.normal(5, 1))
        obstime = obstime + gap
        hdu.header['TIME{:04d}'.format(t)] = obstime.isoformat()

    if verbose is True:
        print(verbose)
        print(metadata)

    base_name = 'PSC_0002'

    hdu.writeto('{}/{}.fits'.format(output_dir, base_name), clobber=True)

    # Generate Fake Lightcurve
    with open('{}/{}.json'.format(output_dir, base_name), 'w') as FO:
        data = []
        for t in range(nt):
            time = hdu.header['TIME{:04d}'.format(t)]
            sig_r = 0.010
            sig_g = 0.006
            sig_b = 0.017
            r = np.random.normal(1, sig_r)
            g = np.random.normal(1, sig_g)
            b = np.random.normal(1, sig_b)
            entry = {
                'Time': time,
                'R': r,
                'G': g,
                'B': b,
                'sig_r': sig_r,
                'sig_g': sig_g,
                'sig_b': sig_b
            }
            data.append(entry)
        json.dump(data, FO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_dir', default='/var/panoptes/PIAA/temp/', help='Where to put generated data')
    parser.add_argument('--verbose', action="store_false", default='False', help='Verbose.')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print("{} does not exist, creating".format(args.output_dir))
        os.mkdir('/var/panoptes/PIAA/temp')

    main(**vars(args))
