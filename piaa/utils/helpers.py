import os
import shutil
import subprocess
from warnings import warn
import google.datalab.storage as storage

import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from astropy.table import Table

from astropy.visualization import LogStretch, ImageNormalize, LinearStretch

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

from photutils import RectangularAnnulus, RectangularAperture

from pocs.utils.images import fits_utils

from decimal import Decimal
from copy import copy

palette = copy(plt.cm.inferno)
palette.set_over('w', 1.0)
palette.set_under('k', 1.0)
palette.set_bad('g', 1.0)

rc('animation', html='html5')
plt.style.use('bmh')


def get_db_conn(instance='panoptes-meta', db='panoptes', **kwargs):
    """ Gets a connection to the Cloud SQL db

    Args:
        instance
    """
    ssl_root_cert = os.path.join(os.environ['SSL_KEYS_DIR'], instance, 'server-ca.pem')
    ssl_client_cert = os.path.join(os.environ['SSL_KEYS_DIR'], instance, 'client-cert.pem')
    ssl_client_key = os.path.join(os.environ['SSL_KEYS_DIR'], instance, 'client-key.pem')
    try:
        pg_pass = os.environ['PGPASSWORD']
    except KeyError:
        warn("DB password has not been set")
        return None

    host_lookup = {
        'panoptes-meta': '146.148.50.241',
        'tess-catalog': '35.226.47.134',
    }

    conn = psycopg2.connect("sslmode=verify-full sslrootcert={} sslcert={} sslkey={} hostaddr={} host=panoptes-survey:{} port=5432 user=postgres dbname={} password={}".format(
        ssl_root_cert, ssl_client_cert, ssl_client_key, host_lookup[instance], instance, db, pg_pass))
    return conn


def get_cursor(with_columns=False, **kwargs):
    conn = get_db_conn(**kwargs)

    cur = conn.cursor(cursor_factory=DictCursor)

    return cur


def meta_insert(table, **kwargs):
    conn = get_db_conn()
    cur = conn.cursor()
    col_names = ','.join(kwargs.keys())
    col_val_holders = ','.join(['%s' for _ in range(len(kwargs))])
    cur.execute('INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING RETURNING *'.format(table,
                                                                                            col_names, col_val_holders), list(kwargs.values()))
    conn.commit()
    try:
        return cur.fetchone()[0]
    except Exception as e:
        print(e)
        return None


def get_stars_from_footprint(wcs_footprint, **kwargs):
    ra = wcs_footprint[:, 0]
    dec = wcs_footprint[:, 1]

    return get_stars(ra.min(), ra.max(), dec.min(), dec.max(), **kwargs)


def get_stars(
        ra_min,
        ra_max,
        dec_min,
        dec_max,
        table='full_catalog',
        cursor_only=True,
        verbose=False,
        *args,
        **kwargs):
    cur = get_cursor(instance='tess-catalog', db='v6', **kwargs)
    cur.execute('SELECT id, ra, dec, tmag, vmag, e_tmag, twomass FROM {} WHERE tmag < 13 AND ra >= %s AND ra <= %s AND dec >= %s AND dec <= %s;'.format(
        table), (ra_min, ra_max, dec_min, dec_max))

    if cursor_only:
        return cur

    d0 = np.array(cur.fetchall())
    if verbose:
        print(d0)
    return Table(
        data=d0,
        names=['id', 'ra', 'dec', 'tmag', 'vmag', 'e_tmag', 'twomass'],
        dtype=['i4', 'f8', 'f8', 'f4', 'f4', 'f4', 'U26'])


def get_star_info(picid=None, twomass_id=None, table='full_catalog', verbose=False, **kwargs):
    cur = get_cursor(instance='tess-catalog', db='v6', **kwargs)
    
    if picid:
        val = picid
        col = 'id'
    elif twomass_id:
        val = twomass_id
        col = 'twomass'
    
    cur.execute('SELECT * FROM {} WHERE {}=%s'.format(table, col), (val,))
    return cur.fetchone()


def get_observation_blobs(prefix=None, key=None, include_pointing=False, project_id='panoptes-survey'):
    """ Returns the list of Google Objects matching the field and sequence """

    # The bucket we will use to fetch our objects
    bucket = storage.Bucket(project_id)
    objs = list()
    
    if prefix:
        for f in bucket.objects(prefix=prefix):
            if 'pointing' in f.key and not include_pointing:
                continue
            elif f.key.endswith('.fz') is False:
                continue
            else:
                objs.append(f)

        return sorted(objs, key=lambda x: x.key)
    
    if key:
        objs = bucket.object(key)
        if objs.exists():
            return objs
        
    return None


def unpack_blob(img_blob, save_dir='/var/panoptes/fits_files/', verbose=False):
    """ Downloads the image blob data, uncompresses, and returns HDU """
    fits_fz_fn = img_blob.key.replace('/', '_')
    fits_fz_fn = os.path.join(save_dir, fits_fz_fn)
    fits_fn = fits_fz_fn.replace('.fz', '')

    if not os.path.exists(fits_fn):
        if verbose:
            print('.', end='')

        download_blob(img_blob, save_as=fits_fz_fn)

    if os.path.exists(fits_fz_fn):
        fits_fn = fits_utils.fpack(fits_fz_fn, unpack=True)

    return fits_fn


def download_blob(img_blob, save_as=None):
    if save_as is None:
        save_as = img_blob.key.replace('/', '_')

    with open(save_as, 'wb') as f:
        f.write(img_blob.download())


def upload_to_bucket(local_path, remote_path, bucket='panoptes-survey', logger=None):
    assert os.path.exists(local_path)

    gsutil = shutil.which('gsutil')
    assert gsutil is not None, "gsutil command line utility not found"

    bucket = 'gs://{}/'.format(bucket)
    # normpath strips the trailing slash so add here so we place in directory
    run_cmd = [gsutil, '-mq', 'cp', local_path, bucket + remote_path]
    if logger:
        logger.debug("Running: {}".format(run_cmd))

    try:
        completed_process = subprocess.run(run_cmd, stdout=subprocess.PIPE)

        if completed_process.returncode != 0:
            if logger:
                logger.debug("Problem uploading")
                logger.debug(completed_process.stdout)
    except Exception as e:
        if logger:
            logger.error("Problem uploading: {}".format(e))


def get_header_from_storage(blob):
    """ Read the FITS header from storage """
    i = 2  # We skip the initial header
    headers = dict()
    while True:
        # Get a header card
        b_string = blob.read_stream(start_offset=2880 * (i - 1), byte_count=(2880 * i) - 1)

        # Loop over 80-char lines
        for j in range(0, len(b_string), 80):
            item_string = b_string[j:j + 80].decode()
            if not item_string.startswith('END'):
                if item_string.find('=') > 0:  # Skip COMMENTS and HISTORY
                    k, v = item_string.split('=')

                    if ' / ' in v:  # Remove FITS comment
                        v = v.split(' / ')[0]

                    v = v.strip()
                    if v.startswith("'") and v.endswith("'"):
                        v = v.replace("'", "").strip()
                    elif v.find('.') > 0:
                        v = float(v)
                    elif v == 'T':
                        v = True
                    elif v == 'F':
                        v = False
                    else:
                        v = int(v)

                    headers[k.strip()] = v
            else:
                return headers
        i += 1


def get_rgb_masks(data, separate_green=False, force_new=False, verbose=False):

    rgb_mask_file = 'rgb_masks.npz'

    if force_new:
        try:
            os.remove(rgb_mask_file)
        except FileNotFoundError:
            pass

    try:
        return np.load(rgb_mask_file)
    except FileNotFoundError:
        if verbose:
            print("Making RGB masks")

        if data.ndim > 2:
            data = data[0]

        w, h = data.shape

        red_mask = np.flipud(np.array(
            [index[0] % 2 == 0 and index[1] % 2 == 0 for index, i in np.ndenumerate(data)]
        ).reshape(w, h))

        blue_mask = np.flipud(np.array(
            [index[0] % 2 == 1 and index[1] % 2 == 1 for index, i in np.ndenumerate(data)]
        ).reshape(w, h))

        if separate_green:
            green1_mask = np.flipud(np.array(
                [(index[0] % 2 == 0 and index[1] % 2 == 1) for index, i in np.ndenumerate(data)]
            ).reshape(w, h))
            green2_mask = np.flipud(np.array(
                [(index[0] % 2 == 1 and index[1] % 2 == 0) for index, i in np.ndenumerate(data)]
            ).reshape(w, h))

            _rgb_masks = np.array([red_mask, green1_mask, green2_mask, blue_mask])
        else:
            green_mask = np.flipud(np.array(
                [(index[0] % 2 == 0 and index[1] % 2 == 1) or (index[0] % 2 == 1 and index[1] % 2 == 0)
                 for index, i in np.ndenumerate(data)]
            ).reshape(w, h))

            _rgb_masks = np.array([red_mask, green_mask, blue_mask])

        _rgb_masks.dump(rgb_mask_file)

        return _rgb_masks


def show_stamps(pscs, frame_idx=None, stamp_size=11, aperture_position=None, aperture_size=None, show_normal=False, show_residual=False, stretch=None, save_name=None, **kwargs):

    if aperture_position is None:
        midpoint = (stamp_size - 1) / 2
        aperture_position = (midpoint, midpoint)

    if aperture_size:
        aperture = RectangularAperture(aperture_position, w=aperture_size, h=aperture_size, theta=0)
        annulus = RectangularAnnulus(aperture_position, w_in=aperture_size, w_out=stamp_size, h_out=stamp_size, theta=0)

    ncols = len(pscs)

    if show_residual:
        ncols += 1

    nrows = 1
    if show_normal:
        nrows = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_dpi(100)
    fig.set_figheight(4)
    fig.set_figwidth(9)

    norm = [normalize(p.reshape(p.shape[0], -1)).reshape(p.shape) for p in pscs]

    if frame_idx is not None:
        s0 = pscs[0][frame_idx]
        n0 = norm[0][frame_idx]

        s1 = pscs[1][frame_idx]
        n1 = norm[1][frame_idx]
    else:
        s0 = pscs[0]
        n0 = norm[0]

        s1 = pscs[1]
        n1 = norm[1]

    if stretch == 'log':
        stretch = LogStretch()
    else:
        stretch = LinearStretch()

    # Target
    if show_normal:
        ax1 = ax[0][0]
    else:
        ax1 = ax[0]
    im = ax1.imshow(s0, origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
    if aperture_size:
        aperture.plot(color='r', lw=4, ax=ax1)
        #annulus.plot(color='c', lw=2, ls='--', ax=ax1)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)        
    ax1.set_title('Target')

    # Normalized target
    if show_normal:
        ax2 = ax[1][0]
        im = ax2.imshow(n0, origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
        aperture.plot(color='r', lw=4, ax=ax2)
        #annulus.plot(color='c', lw=2, ls='--', ax=ax2)
        fig.colorbar(im, ax=ax2)
        ax2.set_title('Normalized Stamp')

    # Comparison
    if show_normal:
        ax1 = ax[0][1]
    else:
        ax1 = ax[1]
    im = ax1.imshow(s1, origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
    if aperture_size:
        aperture.plot(color='r', lw=4, ax=ax1)
        #annulus.plot(color='c', lw=2, ls='--', ax=ax1)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)        
    ax1.set_title('Comparison')
        
    #ax1.set_title('Stamp {:.02f}'.format(get_sum(s1, stamp_size=stamp_size)))

    # Normalized comparison
    if show_normal:
        ax2 = ax[1][1]
        im = ax2.imshow(n1, origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
        aperture.plot(color='r', lw=4, ax=ax2)
        #annulus.plot(color='c', lw=2, ls='--', ax=ax2)
        fig.colorbar(im, ax=ax2)
        ax2.set_title('Normalized Stamp')

    if show_residual:
        if show_normal:
            ax1 = ax[0][2]
        else:
            ax1 = ax[2]

        # Residual
        im = ax1.imshow((s0 / s1), origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
        
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)        
        #ax1.set_title('Residual')
        residual = 1 - (s0.sum() / s1.sum())
        ax1.set_title('Residual {:.01%}'.format(residual))
        
        # Normalized residual
        if show_normal:
            ax2 = ax[1][2]
            im = ax2.imshow((n0 - n1), origin='lower', cmap=palette)
            aperture.plot(color='r', lw=4, ax=ax2)
            #annulus.plot(color='c', lw=2, ls='--', ax=ax2)
            fig.colorbar(im, ax=ax2)
            ax2.set_title('Normalized Stamp')

    #fig.tight_layout()
    
    if save_name:
        try:
            fig.savefig(save_name)
            plt.close(fig)
        except Exception as e:
            warn("Can't save figure: {}".format(e))


def normalize(cube):
    # Helper function to normalize a stamp
    return (cube.T / cube.sum(1)).T


def spiral_matrix(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[:, 0][::-1])  # take first row and reverse it
        A = A[:, 1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)


def pixel_color(x, y):
    """ Given an x,y position, return the corresponding color

    This is a Bayer array with a RGGB pattern in the lower left corner
    as it is loaded into numpy.

    Note:
              0  1  2  3
             ------------
          0 | G2 B  G2 B
          1 | R  G1 R  G1
          2 | G2 B  G2 B
          3 | R  G1 R  G1
          4 | G2 B  G2 B
          5 | R  G1 R  G1

          R : even x, odd y
          G1: odd x, odd y
          G2: even x, even y
          B : odd x, even y

    Returns:
        str: one of 'R', 'G1', 'G2', 'B'
    """
    x = int(x)
    y = int(y)
    if x % 2 == 0:
        if y % 2 == 0:
            return 'G2'
        else:
            return 'R'
    else:
        if y % 2 == 0:
            return 'B'
        else:
            return 'G1'


def get_stamp_slice(x, y, stamp_size=(14, 14), verbose=False):

    for m in stamp_size:
        m -= 2  # Subtract center superpixel
        if int(m / 2) % 2 != 0:
            print("Invalid size: ", m + 2)
            return

    x = Decimal(float(x)).to_integral()
    y = Decimal(float(y)).to_integral()
    color = pixel_color(x, y)
    if verbose:
        print(x, y, color)

    x_half = int(stamp_size[0] / 2)
    y_half = int(stamp_size[1] / 2)

    x_min = int(x - x_half)
    x_max = int(x + x_half)

    y_min = int(y - y_half)
    y_max = int(y + y_half)

    if color == 'B':
        y_min -= 1
        y_max -= 1
    elif color == 'G2':
        x_min -= 1
        x_max -= 1
        y_min -= 1
        y_max -= 1
    elif color == 'R':
        x_min -= 1
        x_max -= 1

    if verbose:
        print(x_min, x_max, y_min, y_max)

    return (slice(y_min, y_max), slice(x_min, x_max))


def animate_stamp(d0):

    fig, ax = plt.subplots()

    line = ax.imshow(d0[0])

    def animate(i):
        line.set_data(d0[i])  # update the data
        return line,

    # Init only required for blitting to give a clean slate.
    def init():
        line.set_data(d0[0])
        return line,

    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(d0)), init_func=init,
                                  interval=500, blit=True)

    return ani
