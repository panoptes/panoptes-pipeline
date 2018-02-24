import os
from astropy.io import fits
import google.datalab.storage as storage

import numpy as np
import psycopg2
from astropy.table import Table

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from astropy.wcs import WCS
from astropy.visualization import (PercentileInterval, LogStretch, ImageNormalize)

import os
import google.datalab.storage as storage
import numpy as np
import numpy.ma as ma

from matplotlib import pyplot as plt
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

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from astropy.wcs import WCS
from astropy.visualization import (PercentileInterval, LogStretch, ImageNormalize)
    
from pocs.utils.images import fits_utils
from pocs.utils import current_time

from copy import copy

palette = copy(plt.cm.inferno)
palette.set_over('w', 1.0)
palette.set_under('k', 1.0)
palette.set_bad('g', 1.0)


def get_stars(ra_min, ra_max, dec_min, dec_max, table='tic', verbose=False):
    ssl_root_cert = os.path.join(os.environ['SSL_KEYS_DIR'], 'server-ca.pem')
    ssl_client_cert = os.path.join(os.environ['SSL_KEYS_DIR'], 'client-cert.pem')
    ssl_client_key = os.path.join(os.environ['SSL_KEYS_DIR'], 'client-key.pem')
    pg_pass = os.environ['PGPASSWORD']
    conn = psycopg2.connect("sslmode=verify-full sslrootcert={} sslcert={} sslkey={} hostaddr=35.226.47.134 host=panoptes-survey:tess-catalog port=5432 user=postgres dbname=v6 password={}".format(ssl_root_cert, ssl_client_cert, ssl_client_key, pg_pass))
    cur = conn.cursor()
    cur.execute('SELECT id, ra, dec, tmag, e_tmag, twomass FROM {} WHERE ra >= %s AND ra <= %s AND dec >= %s AND dec <= %s;'.format(table), (ra_min, ra_max, dec_min, dec_max))
    d0 = np.array(cur.fetchall())
    if verbose:
        print(d0)
    return Table(data=d0, names=['id', 'ra', 'dec', 'tmag', 'e_tmag', 'twomass'], dtype=['i4', 'f8', 'f8', 'f4', 'f4', 'U26'])


def get_observation_blobs(prefix, include_pointing=False, project_id='panoptes-survey'):
  """ Returns the list of Google Objects matching the field and sequence """
  
  # The bucket we will use to fetch our objects
  bucket = storage.Bucket(project_id)    
    
  objs = list()
  for f in bucket.objects(prefix=prefix):
      if 'pointing' in f.key and not include_pointing:
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
    
def get_rgb_masks(data):
  
    rgb_mask_file = 'rgb_masks.numpy'
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

        green_mask = np.flipud(np.array(
            [(index[0] % 2 == 0 and index[1] % 2 == 1) or (index[0] % 2 == 1 and index[1] % 2 == 0)
             for index, i in np.ndenumerate(data)]
        ).reshape(w, h))

        _rgb_masks = np.array([red_mask, green_mask, blue_mask])
        
        _rgb_masks.dump(rgb_mask_file)  
        
        return _rgb_masks    
