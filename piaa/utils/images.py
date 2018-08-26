import os
import subprocess

from copy import copy
from warnings import warn

from matplotlib import pyplot as plt
from astropy.io import fits

from pocs.utils import error
from pocs.utils.images import fits as fits_utils

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
