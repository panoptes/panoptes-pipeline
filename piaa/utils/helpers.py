import os
from warnings import warn
from astropy.io import fits
import google.datalab.storage as storage

import numpy as np
import psycopg2
from astropy.table import Table

from astropy.visualization import SqrtStretch
from astropy.visualization import LogStretch, ImageNormalize, LinearStretch
from photutils import CircularAperture
from astropy.wcs import WCS
from astropy.visualization import (PercentileInterval, LogStretch, ImageNormalize)

from scipy.optimize import minimize
from scipy.sparse.linalg import lsmr, lsqr

import os
import google.datalab.storage as storage
import numpy as np
import numpy.ma as ma

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='html5') 

plt.style.use('bmh')

from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy import units as u
from astropy import constants as c

import photutils
from photutils import Background2D, MeanBackground, MMMBackground, \
    MedianBackground, SExtractorBackground, BkgIDWInterpolator, BkgZoomInterpolator, \
    make_source_mask
    
from photutils import DAOStarFinder
from photutils import find_peaks, RectangularAnnulus, RectangularAperture, aperture_photometry

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from astropy.wcs import WCS
from astropy.visualization import (PercentileInterval, LogStretch, ImageNormalize)
    
from pocs.utils.images import fits_utils
from pocs.utils import current_time

from decimal import Decimal
from copy import copy

palette = copy(plt.cm.inferno)
palette.set_over('w', 1.0)
palette.set_under('k', 1.0)
palette.set_bad('g', 1.0)


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
        
    conn = psycopg2.connect("sslmode=verify-full sslrootcert={} sslcert={} sslkey={} hostaddr={} host=panoptes-survey:{} port=5432 user=postgres dbname={} password={}".format(ssl_root_cert, ssl_client_cert, ssl_client_key, host_lookup[instance], instance, db, pg_pass))
    return conn

def get_cursor(**kwargs):
    conn = get_db_conn(**kwargs)
    cur = conn.cursor()
    
    return cur

def meta_insert(table, **kwargs):    
    conn = get_db_conn()
    cur = conn.cursor()
    col_names = ','.join(kwargs.keys())
    col_val_holders = ','.join(['%s' for _ in range(len(kwargs))])
    cur.execute('INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING RETURNING *'.format(table, col_names, col_val_holders), list(kwargs.values()))
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

def get_stars(ra_min, ra_max, dec_min, dec_max, table='full_catalog', cursor_only=True, verbose=False, *args, **kwargs):
    cur = get_cursor(instance='tess-catalog', db='v6')
    cur.execute('SELECT id, ra, dec, tmag, e_tmag, twomass FROM {} WHERE tmag < 13 AND ra >= %s AND ra <= %s AND dec >= %s AND dec <= %s;'.format(table), (ra_min, ra_max, dec_min, dec_max))
    
    if cursor_only:
        return cur
    
    d0 = np.array(cur.fetchall())
    if verbose:
        print(d0)
    return Table(data=d0, names=['id', 'ra', 'dec', 'tmag', 'e_tmag', 'twomass'], dtype=['i4', 'f8', 'f8', 'f4', 'f4', 'U26'])

def get_star_info(twomass_id, table='full_catalog', verbose=False):
    cur = get_cursor(instance='tess-catalog', db='v6')
    
    cur.execute('SELECT * FROM {} WHERE twomass=%s'.format(table), (twomass_id,))
    d0 = np.array(cur.fetchall())
    if verbose:
        print(d0)
    return d0


def get_observation_blobs(prefix, include_pointing=False, project_id='panoptes-survey'):
  """ Returns the list of Google Objects matching the field and sequence """
  
  # The bucket we will use to fetch our objects
  bucket = storage.Bucket(project_id)    
    
  objs = list()
  for f in bucket.objects(prefix=prefix):
      if 'pointing' in f.key and not include_pointing:
        continue
      elif f.key.endswith('.fz') is False:
        continue
      else:
        objs.append(f)
        
  return sorted(objs, key=lambda x: x.key)

def unpack_blob(img_blob, save_dir='/var/panoptes/fits_files/', remove_file=False):
  """ Downloads the image blob data, uncompresses, and returns HDU """
  fits_fz_fn = img_blob.key.replace('/', '_')
  fits_fz_fn = os.path.join(save_dir, fits_fz_fn)
  fits_fn = fits_fz_fn.replace('.fz', '')
  
  if not os.path.exists(fits_fn):
    print('.', end='')
    with open(fits_fz_fn, 'wb') as f:
        f.write(img_blob.download())

  if os.path.exists(fits_fz_fn):
    fits_fn = fits_utils.fpack(fits_fz_fn, unpack=True)

  return fits_fn
  

def get_header(blob):
  """ Read the FITS header from storage """
  i = 2 # We skip the initial header
  headers = dict()
  while True:
    # Get a header card
    b_string = blob.read_stream(start_offset=2880 * (i - 1), byte_count=(2880 * i) - 1)

    # Loop over 80-char lines
    for j in range(0, len(b_string), 80):
      item_string = b_string[j:j+80].decode()
      if not item_string.startswith('END'):
        if item_string.find('=') > 0: # Skip COMMENTS and HISTORY
          k, v = item_string.split('=')
          
          if ' / ' in v: # Remove FITS comment
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
    
    
def get_rgb_masks(data, separate_green=False, force_new=False):
    
    rgb_mask_file = 'rgb_masks.npz'
    
    if force_new:
        try:
            os.remove(rgb_mask_file)
        except FileNotFoundError:
            pass
  
    try:
        return np.load(rgb_mask_file)
    except FileNotFoundError:
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

def get_psc(idx=None, ticid=None, aperture_size=None, get_masks=False, stamp_size=11, stamp_dir=None, stamp_cubes=None, verbose=False):
    if idx is not None:
        d0 = np.load(stamp_cubes[idx])
        
    if ticid is not None:
        d0 = np.load(os.path.join(stamp_dir, '{}.npz'.format(ticid)))
    
    psc = d0['psc']
    pos = d0['pos']
    if verbose:
        print(pos)

    midpoint = int((stamp_size-1)/2)
    
    masks = list()
    if get_masks:
        if aperture_size is not None:
            size = aperture_size
        else:
            size = stamp_size
        for color, mask in rgb_masks.items():
            masks.append(np.array([Cutout2D(mask, p, size, mode='strict').data.flatten() for p in pos]))
    else:    
        if aperture_size is not None:
            psc = np.array([Cutout2D(s.reshape(stamp_size, stamp_size), (midpoint,midpoint), aperture_size, mode='strict').data.flatten() for s in psc])
    
    if get_masks is False:
        return psc
    else:
        return np.array(masks)

def show_stamps(idx_list=None, pscs=None, frame_idx=0, stamp_size=11, aperture_size=4, show_residual=False, stretch=None, **kwargs):
    
    midpoint = (stamp_size - 1) / 2
    aperture = RectangularAperture((midpoint, midpoint), w=aperture_size, h=aperture_size, theta=0)
    annulus = RectangularAnnulus((midpoint, midpoint), w_in=aperture_size, w_out=stamp_size, h_out=stamp_size, theta=0)    
    
    if idx_list is not None:
        pscs = [get_psc(i, stamp_size=stamp_size, **kwargs) for i in idx_list]
        ncols = len(idx_list)
    else:
        ncols = len(pscs)
    
    if show_residual:
        ncols += 1
    
    fig, ax = plt.subplots(nrows=2, ncols=ncols)
    fig.set_figheight(6)
    fig.set_figwidth(12)

    norm = [normalize(p.reshape(p.shape[0], -1)).reshape(p.shape) for p in pscs]

    s0 = pscs[0][frame_idx]
    n0 = norm[0][frame_idx]
    
    s1 = pscs[1][frame_idx]
    n1 = norm[1][frame_idx]    
        
    if stretch == 'log':
        stretch = LogStretch()
    else:
        stretch = LinearStretch()       
        
    # Target
    ax1 = ax[0][0]
    im = ax1.imshow(s0, origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax1)
    annulus.plot(color='c', lw=2, ls='--', ax=ax1)
    fig.colorbar(im, ax=ax1)
    #ax1.set_title('Stamp {:.02f}'.format(get_sum(s0, stamp_size=stamp_size)))

    # Normalized target
    ax2 = ax[1][0]
    im = ax2.imshow(n0, origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax2)
    annulus.plot(color='c', lw=2, ls='--', ax=ax2)
    fig.colorbar(im, ax=ax2)
    ax2.set_title('Normalized Stamp')
        
    # Comparison
    ax1 = ax[0][1]
    im = ax1.imshow(s1, origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax1)
    annulus.plot(color='c', lw=2, ls='--', ax=ax1)
    fig.colorbar(im, ax=ax1)
    #ax1.set_title('Stamp {:.02f}'.format(get_sum(s1, stamp_size=stamp_size)))

    # Normalized comparison
    ax2 = ax[1][1]
    im = ax2.imshow(n1, origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
    aperture.plot(color='r', lw=4, ax=ax2)
    annulus.plot(color='c', lw=2, ls='--', ax=ax2)
    fig.colorbar(im, ax=ax2)
    ax2.set_title('Normalized Stamp')        
        
    if show_residual:

        # Residual
        ax1 = ax[0][2]
        im = ax1.imshow((s0 - s1), origin='lower', cmap=palette, norm=ImageNormalize(stretch=stretch))
        aperture.plot(color='r', lw=4, ax=ax1)
        annulus.plot(color='c', lw=2, ls='--', ax=ax1)
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Stamp Residual - {:.02f}'.format((s0 - s1).sum()))

        # Normalized residual
        ax2 = ax[1][2]
        im = ax2.imshow((n0 - n1), origin='lower', cmap=palette)
        aperture.plot(color='r', lw=4, ax=ax2)
        annulus.plot(color='c', lw=2, ls='--', ax=ax2)
        fig.colorbar(im, ax=ax2)
        ax2.set_title('Normalized Stamp')                
        
    fig.tight_layout()

# Helper function to normalize a stamp
def normalize(cube):
    return (cube.T / cube.sum(1)).T

def get_stamp_difference(d0, d1):
    return ((d0 - d1)**2).sum()

def spiral_matrix(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[:,0][::-1])  # take first row and reverse it
        A = A[:,1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)

def get_ideal_full_coeffs(stamp_collection, damp=1, func=lsmr, verbose=False):

    num_refs = stamp_collection.shape[0] - 1
    num_frames = stamp_collection.shape[1]
    num_pixels = stamp_collection.shape[2]
    
    target_frames = stamp_collection[0].flatten()
    refs_frames = stamp_collection[1:].reshape(-1, num_frames * num_pixels).T
                    
    if verbose:
        print("Target other shape: {}".format(target_frames.shape))
        print("Refs other shape: {}".format(refs_frames.shape))        
    
    coeffs = func(refs_frames, target_frames, damp)
    
    return coeffs

def get_ideal_full_psc(stamp_collection, coeffs, **kwargs):        

    num_frames = stamp_collection.shape[1]

    refs = stamp_collection[1:]
        
    created_frame = (refs.T * coeffs).sum(2).T

    return created_frame

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

def superpixel_position(x, y):
    """ Given an x,y coordinate, return the x,y corresponding to the red superpixel position """
    x = int(x)
    y = int(y)
    color = pixel_color(x, y)
    if color == 'R':
        return x, y
    elif color == 'G1':
        return x-1, y
    elif color == 'G2':
        return x, y+1
    elif color == 'B':
        return x-1, y+1

def get_cutout_position(x, y):
    """ Convenience function to nudge a superpixel position to correct cutout position """
    super_x, super_y = superpixel_position(x, y)
    return super_x, super_y

def get_stamp_slice(x, y, stamp_size=(6, 6), verbose=False):
    width = stamp_size[0]
    height = stamp_size[1]
    
    for m in stamp_size:
        m -= 2 # Subtract center superpixel
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
        
    return [slice(y_min, y_max), slice(x_min, x_max)]
        
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