import os

from astropy import units as u
from astropy.time import Time

from collections import namedtuple

import batman

# Query Exoplanet Orbit Database (exoplanets.org) for planet properties
# Columns:http://exoplanets.org/help/common/data
from astroquery.exoplanet_orbit_database import ExoplanetOrbitDatabase
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
    }
}


class Exoplanet():

    def __init__(self, name, db='exoplanet_orbit_database', verbose=False, *args, **kwargs):
        self.verbose = verbose
        self.name = name

        self._print("Looking up info for {}".format(self.name))

        try:
            self._keymap = EXOPLANET_DB_KEYMAP[db]['keymap']
            self.info = EXOPLANET_DB_KEYMAP[db]['query_method'](name)
        except KeyError:
            raise Exception("No exoplanet DB called {}".format(db))
        else:
            assert self.info is not None
            self._db = db
            self._loookups = dict()

        # Get the transit system for calculating ephemris
        self.transit_system = EclipsingSystem(
            primary_eclipse_time=self.midtransit,
            orbital_period=self.period,
            duration=self.transit_duration
        )

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
        return Time(self.get_prop('midtransit'), format='jd')

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
        semimajor_axis = self.info['A'].to(u.R_sun)
        eccentricity = self.info['ECC']
        planet_radius = self.info['R'].to(u.R_sun)
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

        transit_params.a = semimajor_axis.value  # semi-major axis (stellar radii)
        transit_params.ecc = eccentricity
        transit_params.w = periastron.value  # longitude of periastron (in degrees)

        transit_params.limb_dark = "uniform"  # limb darkening model
        transit_params.u = []  # limb darkening coefficients [u1, u2, u3, u4]

        return transit_params

    def get_model_lightcurve(self, index, period=None):
        """Gets the model lightcurve.

        Args:
            index (list or `numpy.array`): The index to be used, can either be a
                list of time objects or an array of phases.
            period (float, optional): The period passed to `get_model_params`.

        Returns:
            `numpy.array`: An array of normalized flux values.
        """
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
            ing_egr.datetime[0][0],
            next_transit[0].datetime,
            ing_egr.datetime[0][1]
        )

        return transit_info
