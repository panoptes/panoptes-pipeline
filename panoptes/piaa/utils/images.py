import os
import subprocess
from warnings import warn

from astropy.io import fits

from panoptes.utils import error
from panoptes.utils.images import fits as fits_utils


def improve_wcs(fname, remove_extras=True, replace=True, timeout=30, **kwargs):
    """Improve the world-coordinate-system (WCS) of a FITS file.

    This will plate-solve an already-solved field, using a verification process
    that will also attempt a SIP distortion correction.

    Args:
        fname (str): Full path to FITS file.
        remove_extras (bool, optional): If generated files should be removed, default True.
        replace (bool, optional): Overwrite existing file, default True.
        timeout (int, optional): Timeout for the solve, default 30 seconds.
        **kwargs: Additional keyword args for `solve_field`. Can also include a
            `verbose` flag.

    Returns:
        dict: FITS headers, including solve information.

    Raises:
        error.SolveError: Description
        error.Timeout: Description
    """
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
