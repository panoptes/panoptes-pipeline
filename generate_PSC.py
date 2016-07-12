from datetime import datetime as dt
from datetime import timedelta as tdelta
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

sky_background = 1000.
sky_sigma = 5.
nx = 12
ny = 16
nt = 42
data_cube = np.random.normal(sky_background, sky_sigma, (nt,ny,nx))
obstime = dt.utcnow()
unit = 'PAN000' #I've given this the ID of PAN000 just in case it gets confused for data from a real unit.
camera = '0x2A'
target_name = 'faketarg'
seq_id = '{}{}_{}'.format(unit, camera, obstime.strftime('%Y%m%d_%H%M%SUT'))
xpixorg = 1042
ypixorg = 2042
exptime = 100. # seconds
c = SkyCoord.from_name('HR8799', frame='fk5')

hdu = fits.PrimaryHDU(data_cube)
metadata = {'SEQID': seq_id,\
            'FIELD': target_name,\
            'RA': c.ra.to(u.degree).value,\
            'DEC': c.dec.to(u.degree).value,\
            'EQUINOX': c.equinox.value,\
            'OBSTIME': obstime.isoformat(),\
            'XPIXORG': xpixorg,\
            'YPIXORG': ypixorg,\
            }

for t in range(nt):
    # slightly randomize time gap between images
    gap = tdelta(0,exptime + np.random.normal(5,1))
    obstime = obstime + t*gap
    metadata['TIME{:04d}'.format(t)] = obstime.isoformat()
hdu.header.extend(metadata)

hdu.writeto('PSC_0001.fits', clobber=True)
