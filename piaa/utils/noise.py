import logging
import numpy as np

from astropy.stats import sigma_clipped_stats
from astropy import units as u

from piaa.utils import helpers

logger = logging.getLogger(__name__)


def get_stamp_noise(
        stamp,
        exptime,           # seconds
        camera_bias=2048,  # ADU
        gain=1.5,          # e- / ADU
        readout_noise=10.5, # e-
        return_detail=False
    ):
    """Gets the noise for a stamp computed from given values. """

    if hasattr(stamp, 'mask'):
        num_pixels = np.count_nonzero(~stamp.mask)
    else:
        num_pixels = int(stamp.shape[0] * stamp.shape[1])

    # Remove built-in bias
    stamp_counts = stamp - camera_bias

    # Convert to electrons with gain
    stamp_electrons = stamp_counts * gain
    
    # Get the sigma-clipped stats for the electrons, i.e. the background
    back_mean, back_median, back_std = sigma_clipped_stats(stamp_electrons)
    
    # Get background subtracted electrons
    electron_sum = (stamp_electrons - back_mean).sum()

    if electron_sum <= 0:
        raise ValueError("Negative electrons found")

    # Photon noise
    photon_noise = np.sqrt(electron_sum)
    
    # Readout noise
    readout = readout_noise * num_pixels
    
    # Dark noise (for Canon DSLR, see Zhang et al 2016)
    dark_noise = int(0.1 * exptime) * num_pixels

    noise_sum = np.sqrt(
        photon_noise**2 +
        back_std**2 +
        readout_noise**2 +
        dark_noise**2
    )

    # Convert electrons back to counts
    count_sum = electron_sum / gain
    photon_noise /= gain
    back_mean /= gain
    back_noise = back_std / gain
    readout /= gain
    dark_noise /= gain
    noise_sum /= gain
    
    noises = {
        'counts': count_sum,
        'noise': noise_sum,
    }

    if return_detail:
        noises.update({
            'photon_noise': photon_noise,
            'back': back_mean,
            'back_noise': back_noise,
            'readout_noise': readout,
            'dark_noise': dark_noise,
        })

    return noises


def estimated_photon_count(
        magnitude=0,
        aperture_area=1, 
        airmass=1, 
        filter_name='V', 
        qe=1.,
        ):

    if magnitude is None:
        return None
    
    if not isinstance(aperture_area, u.Quantity):
        aperture_area *= (u.m * u.m)

    flux_params = helpers.get_photon_flux_params(filter_name)
    logger.info(f'Using flux params: {flux_params}')
    
    flux0 = flux_params['flux0'] #* u.jansky
    extinction = flux_params['extinction']
    filter_center = flux_params['lambda_c'] * u.micron
    dlambda_ratio = flux_params['dlambda_ratio']
    filter_width = flux_params['filter_width'] * u.nm

    logger.info(f'Initial flux: {flux0:.02f} J')

    # Adjust for magnitude (scales magnitude)
    flux0 = 10**(-0.4 * magnitude)  * flux0
    logger.info(f'Magnitude scaled ({filter_name}={magnitude}) flux: {flux0:.02f} J')
    flux0 *= 1.51e7 * dlambda_ratio * aperture_area.to(u.m**2)
    flux0 = flux0.value
    logger.info(f'Magnitude scaled flux: {flux0:.02f} photons')

    # Get initial instrumental magnitude
    imag0 = -2.5 * np.log10(flux0)
    logger.info(f'Initial inst mag flux: {imag0:.02f}')

    # Atmosphere causes flux reduction (adds magnitude)
    imag0 += extinction * airmass
    logger.info(f'Airmass corrected (X={airmass:.02f}) inst mag: {imag0:.02f}')

    # Convert back to photons
    photon1 = 10**(imag0 / -2.5) #/ (u.cm * u.cm) / (u.angstrom)
    logger.info(f'Corrected photons: {photon1:.02f}')

    # Quantum efficiency of detector (limit what is detected)
    photon1 *= qe
    logger.info(f'QE ({qe:.0%}) photons: {photon1:.02f}')
         
    return photon1

