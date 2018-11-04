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
    
    flux0 = flux_params['flux0'] * u.jansky
    photon0 = flux_params['photon0'] * u.photon #/ (u.cm * u.cm) / (u.angstrom)
    extinction = flux_params['extinction']
    filter_center = flux_params['lambda_c'] * u.micron
    dlambda_ratio = flux_params['dlambda_ratio']
    filter_width = flux_params['filter_width'] * u.nm

    logger.info(f'Initial photons: {photon0:.02f}')

    # Adjust for magnitude (scales magnitude)
    photon0 *= 10**(-0.4 * magnitude)
    logger.info(f'Magnitude scaled photons: {photon0:.02f}')

    # Get initial instrumental magnitude
    imag0 = -2.5 * np.log10(photon0.value)
    logger.info(f'Inital inst mag: {imag0:.02f}')

    # Atmosphere causes flux reduction (adds magnitude)
    imag0 += extinction * airmass
    logger.info(f'Atmo corrected inst mag: {imag0:.02f}')

    # Convert back to photons
    photon1 = 10**(imag0 / -2.5) / (u.cm * u.cm) / (u.angstrom)
    logger.info(f'Corrected photons: {photon1:.02f}')

    # Multiply by the filter width (which photons get through)
    photon1 *= filter_width.to(u.angstrom)
    logger.info(f'Filter width photons: {photon1:.02f}')

    # Multiply by collecting area (collect more photons)
    logger.info(f'Aperture area: {aperture_area:0.4f}')
    photon1 *= aperture_area.to(u.cm * u.cm)
    logger.info(f'Area collected photons: {photon1:.02f}')

    # Quantum efficiency of detector (limit what is detected)
    photon1 *= qe
    logger.info(f'QE photons: {photon1:.02f}')
         
    return photon1

