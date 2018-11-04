import logging
import numpy as np
from astropy import units as u

from piaa.utils import helpers

logger = logging.getLogger(__name__)

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

