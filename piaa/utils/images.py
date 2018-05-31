import os
import subprocess

from warnings import warn

from matplotlib import pyplot as plt

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import LogStretch, ImageNormalize, LinearStretch, SqrtStretch
from photutils import CircularAperture
from astropy.wcs import WCS
from astropy.visualization import (PercentileInterval, LogStretch, ImageNormalize)

from pocs.utils import error, current_time

from pocs.utils.images import fits as fits_utils

from copy import copy
palette = copy(plt.cm.inferno)
palette.set_over('w', 1.0)
palette.set_under('k', 1.0)
palette.set_bad('g', 1.0)


def improve_wcs(fname, remove_extras=True, replace=True, timeout=30, **kwargs):
    verbose = kwargs.get('verbose', False)
    out_dict = {}
    output = None
    errs = None

    if verbose:
        print("Entering improve_wcs: {}".format(fname))

    options = [
        '--continue',
        '-t', '3',
        '-q', '0.01',
        '--no-plots',
        '--guess-scale',
        '--cpulimit', str(timeout),
        '--no-verify',
        '--crpix-center',
        '--match', 'none',
        '--corr', 'none',
        '--wcs', 'none',
        '-V', fname,
    ]

    proc = fits_utils.solve_field(fname, solve_opts=options, **kwargs)
    try:
        output, errs = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise error.Timeout("Timeout while solving")
    else:
        if verbose:
            print("Output: {}", output)
            print("Errors: {}", errs)

        if not os.path.exists(fname.replace('.fits', '.solved')):
            raise error.SolveError('File not solved')

        try:
            # Handle extra files created by astrometry.net
            new = fname.replace('.fits', '.new')
            rdls = fname.replace('.fits', '.rdls')
            axy = fname.replace('.fits', '.axy')
            xyls = fname.replace('.fits', '-indx.xyls')

            if replace and os.path.exists(new):
                # Remove converted fits
                os.remove(fname)
                # Rename solved fits to proper extension
                os.rename(new, fname)

                out_dict['solved_fits_file'] = fname
            else:
                out_dict['solved_fits_file'] = new

            if remove_extras:
                for f in [rdls, xyls, axy]:
                    if os.path.exists(f):
                        os.remove(f)

        except Exception as e:
            warn('Cannot remove extra files: {}'.format(e))

    if errs is not None:
        warn("Error in solving: {}".format(errs))
    else:
        try:
            out_dict.update(fits.getheader(fname))
        except OSError:
            if verbose:
                print("Can't read fits header for {}".format(fname))

    return out_dict

def make_pretty_from_fits(header, data, figsize=(10, 8), dpi=150, alpha=0.2, pad=3.0, **kwargs):
    wcs = WCS(header)
    data = np.ma.array(data, mask=(data > 12000))
    
    title = kwargs.get('title', header.get('FIELD', 'Unknown'))
    exp_time = header.get('EXPTIME', 'Unknown')

    filter_type = header.get('FILTER', 'Unknown filter')
    date_time = header.get('DATE-OBS', current_time(pretty=True)).replace('T', ' ', 1)

    percent_value = kwargs.get('normalize_clip_percent', 99.9)

    title = '{} ({}s {}) {}'.format(title, exp_time, filter_type, date_time)
    norm = ImageNormalize(interval=PercentileInterval(percent_value), stretch=LogStretch())

    plt.figure(figsize=figsize, dpi=dpi)

    if wcs.is_celestial:
        ax = plt.subplot(projection=wcs)
        ax.coords.grid(True, color='white', ls='-', alpha=alpha)

        ra_axis = ax.coords['ra']
        dec_axis = ax.coords['dec']

        ra_axis.set_axislabel('Right Ascension')
        dec_axis.set_axislabel('Declination')

        ra_axis.set_major_formatter('hh:mm')
        dec_axis.set_major_formatter('dd:mm')

        ra_axis.set_ticks(spacing=5 * u.arcmin, color='white', exclude_overlapping=True)
        dec_axis.set_ticks(spacing=5 * u.arcmin, color='white', exclude_overlapping=True)

        ra_axis.display_minor_ticks(True)
        dec_axis.display_minor_ticks(True)

        dec_axis.set_minor_frequency(10)
    else:
        ax = plt.subplot()
        ax.grid(True, color='white', ls='-', alpha=alpha)

        ax.set_xlabel('X / pixels')
        ax.set_ylabel('Y / pixels')

    ax.imshow(data, norm=norm, cmap=palette, origin='lower')

    plt.tight_layout(pad=pad)
    plt.title(title)

    new_filename = 'pretty.png'
    plt.savefig(new_filename)
#     plt.show()

    plt.close()    