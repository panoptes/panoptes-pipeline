import os

from warnings import warn
import numpy as np

from astropy import units as u
from astropy.time import Time
from astropy.table import Column

from collections import namedtuple

import batman
from pocs.utils import listify

# Query Exoplanet Orbit Database (exoplanets.org) for planet properties
# Columns: http://exoplanets.org/help/common/data
from astroquery.exoplanet_orbit_database import ExoplanetOrbitDatabase
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

from astroplan import EclipsingSystem


TransitInfo = namedtuple('TransitInfo', ['ingress', 'midpoint', 'egress'])


def get_exotable(name):
    from astropy.table import Table
    planets_file = os.path.join(os.environ['PANDIR'], 'PIAA', 'resources', 'planets.csv')
    assert os.path.exists(planets_file)

    # TODO convert to pd.read_csv
    exo_table = Table.read(planets_file, format='ascii.csv',
                           comment='#', header_start=0, data_start=1).to_pandas()

    # Make planet names that match ours
    exo_table['clean_name'] = [row.title().replace(' ', '').replace('-', '')
                               for row in exo_table['pl_hostname']]
    exo_table.set_index(['clean_name'], inplace=True)

    return exo_table.loc[name]


EXOPLANET_DB_KEYMAP = {
    'exoplanet_orbit_database': {
        'query_method': ExoplanetOrbitDatabase.query_planet,
        'keymap': {
            'transit_duration': 'T14',
            'transit_depth': 'DEPTH',
            'period': 'PER',
            'period_ref': 'PERREF',
            'midtransit': 'TT',
            'midtransit_ref': 'TTREF',
            'star_mag': 'V',
        }
    },
    'exotable': {
        'query_method': get_exotable,
        'keymap': {
            'transit_duration': 'pl_trandur',
            'transit_depth': 'pl_trandep',
            'period': 'pl_orbper',
            'period_ref': '',
            'midtransit': 'pl_tranmid',
            'midtransit_ref': '',
            'star_mag': 'st_optmag',
        }
    },
    'nasa': {
        'query_method': NasaExoplanetArchive.query_planet,
        'keymap': {
            'transit_duration': 'pl_trandur',  # Days
            'transit_depth': 'pl_trandep',  # Percentage
            'period': 'pl_orbper',  # Days
            'period_ref': '',
            'midtransit': 'pl_tranmid',  # Julian Days
            'midtransit_ref': '',
            'star_mag': 'st_optmag',  # V Mag
        }
    }
}


class Exoplanet():

    def __init__(self, name, db='nasa', verbose=False, transit_system=None, *args, **kwargs):
        self.verbose = verbose
        self.name = name

        self._print("Looking up info for {}".format(self.name))

        # Bug with all_columns means we need to download table first
        if db == 'nasa':
            NasaExoplanetArchive.get_confirmed_planets_table(all_columns=True)

        try:
            db_map = EXOPLANET_DB_KEYMAP[db]
        except KeyError:
            raise Exception("No exoplanet DB called {}".format(db))

        try:
            self._keymap = db_map['keymap']
            self.info = db_map['query_method'](name)
        except KeyError:
            raise Exception("No exoplanet {}".format(name))
        else:
            assert self.info is not None
            self._db = db
            self._loookups = dict()

        self.transit_system = transit_system

        if not self.transit_system:
            # Get the transit system for calculating ephemris
            try:
                self.transit_system = EclipsingSystem(
                    primary_eclipse_time=self.midtransit,
                    orbital_period=self.period,
                    duration=self.transit_duration
                )
            except Exception:
                pass

    def get_prop(self, col, raw=False):
        val = None

        if raw is False:
            try:
                table_col = self._keymap[col]
            except KeyError:
                self._print("Invalid property: {}".format(col))
                return None
        else:
            table_col = col

        try:
            # Try info in table
            val = self.info[table_col]
        except KeyError:
            self._print("Can't find {} in table".format(table_col))

        return val

    @property
    def transit_duration(self):
        """ """
        return self.get_prop('transit_duration')

    @property
    def period(self):
        """ """
        return self.get_prop('period')

    @property
    def midtransit(self):
        """ """
        try:
            return Time(self.get_prop('midtransit'), format='jd')
        except ValueError:
            return None

    @property
    def star_mag(self):
        """ """
        return self.get_prop('star_mag')

    def _print(self, msg, *args, **kwargs):
        if self.verbose:
            print(msg, *args, **kwargs)

    def get_model_params(self, period=None):
        """Gets the model parameters for known transit.

        Uses the looked up parameters for the exoplanet to populate a set of
        parmaeters for modelling a transit via the `batman` module.

        https://www.cfa.harvard.edu/~lkreidberg/batman/index.html

        Args:
            period (None, optional): If given, should be the period in days. If
                None, use a phase of -0.5 to 0.5 with midpoint at 0.0.

        Returns:
            `batman.TransitParams`: The parameters for use in calcualting a
                transit.
        """
        semimajor_axis = self.info['AR']
        eccentricity = self.info['ECC']
        planet_radius = self.info['R'].to(u.R_sun) / self.info['RSTAR']
        orbital_inc = self.info['I']
        periastron = self.info['OM']

        transit_params = batman.TransitParams()  # object to store transit parameters

        transit_params.t0 = 0.  # time of inferior conjunction

        if period:
            transit_params.per = self.period.value
        else:
            transit_params.per = 1

        transit_params.rp = planet_radius.value  # planet radius (stellar radii)
        transit_params.inc = orbital_inc.value  # orbital inclination (degrees)

        transit_params.a = semimajor_axis  # semi-major axis (stellar radii)
        transit_params.ecc = eccentricity
        transit_params.w = periastron.value  # longitude of periastron (in degrees)

        transit_params.limb_dark = "uniform"  # limb darkening model
        transit_params.u = []  # limb darkening coefficients [u1, u2, u3, u4]

        return transit_params

    def get_model_lightcurve(self, index, period=None, transit_params=None):
        """Gets the model lightcurve.

        Args:
            index (list or `numpy.array`): The index to be used, can either be a
                list of time objects or an array of phases.
            period (float, optional): The period passed to `get_model_params`.

        Returns:
            `numpy.array`: An array of normalized flux values.
        """
        if not transit_params:
            transit_params = self.get_model_params(period=period)

        transit_model = batman.TransitModel(transit_params, index)

        # Get model flux
        model_flux = transit_model.light_curve(transit_params)

        return model_flux

    def get_transit_info(self, obstime=None):
        """Return the next transit information after obstime.

        Args:
            obstime (`astropy.time.Time`, optional): Time seeking next transit for.

        Returns:
            `TransitInfo`: A namedtuple with `ingress`, `midpoint`, and `egress`
                attributes.
        """
        # Calculate next transit times which occur after first image
        if obstime is None:
            obstime = Time.now()

        next_transit = self.transit_system.next_primary_eclipse_time(obstime)
        ing_egr = self.transit_system.next_primary_ingress_egress_time(obstime)

        # Get transit properties
        transit_info = TransitInfo(
            Time(ing_egr.datetime[0][0]),
            Time(next_transit[0].datetime),
            Time(ing_egr.datetime[0][1])
        )

        return transit_info

    def in_transit(self, obstime=None, with_times=False):
        if obstime is None:
            obstime = Time.now()
        obstime = listify(obstime)

        ing_egr = self.transit_system.next_primary_ingress_egress_time(Time(obstime))

        time_checks = list()
        for t in obstime:
            time_in = Time(t) > ing_egr[0][0] and Time(t) < ing_egr[0][1]
            time_checks.append(time_in)

        if with_times:
            return any(time_checks), ing_egr

        return any(time_checks)

    def get_phase(self, obstime, transit_times=None):
        """Get the phase for given time.

        Args:
            obstime (`astropy.time.Time`): Time.

        Returns:
            float: Phase in range [-0.5,0.5] where 0.0 is the midpoint of tranist.
        """
        if not transit_times:
            transit_times = self.get_transit_info(obstime)

        phase = (transit_times.midpoint - obstime).value / self.period.value

        if phase > 0.5:
            phase -= 1.0

        if phase < -0.5:
            phase += 1.0

        return phase


def get_exoplanet_transit(field, index, transit_times=None, model_params=None):
    exoplanet = Exoplanet(field)

    if not transit_times:
        transit_times = TransitInfo(
            Time('2018-08-22 04:53:00'),
            Time('2018-08-22 05:47:00'),
            Time('2018-08-22 06:41:00')
        )

    if not model_params:
        model_params = {
            'a': 8.83602,
            'ecc': 0.0,
            'fp': None,
            'inc': 85.71,
            'limb_dark': 'uniform',
            'per': 2.21857567,
            'rp': 0.15468774550850156,
            't0': 0.0,
            't_secondary': None,
            'u': list(),
            'w': 90.0,
        }

    transit_params = batman.TransitParams()

    transit_params.t0 = 0.  # time of inferior conjunction
    transit_params.per = model_params['per']

    transit_params.rp = model_params['rp']  # planet radius (stellar radii)
    transit_params.inc = model_params['inc']  # orbital inclination (degrees)

    transit_params.a = model_params['a']  # semi-major axis (stellar radii)
    transit_params.ecc = model_params['ecc']
    transit_params.w = model_params['w']  # longitude of periastron (in degrees)

    transit_params.limb_dark = "uniform"  # limb darkening model
    transit_params.u = []  # limb darkening coefficients [u1, u2, u3, u4]

    # Get the orbital phase of the exoplanet
    dt = np.array([(transit_times.midpoint - Time(t0)).value for t0 in index])
    transit_model = batman.TransitModel(transit_params, dt)

    base_model_flux = transit_model.light_curve(transit_params)

    return base_model_flux
