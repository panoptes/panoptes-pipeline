import os

import pandas as pd

# Query Exoplanet Orbit Database (exoplanets.org) for planet properties
# Columns:http://exoplanets.org/help/common/data
from astroquery.exoplanet_orbit_database import ExoplanetOrbitDatabase
from astroplan import EclipsingSystem


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

    def get_prop(self, col):
        val = None

        try:
            table_col = self._keymap[col]
        except KeyError:
            self._print("Invalid property: {}".format(col))
            return None

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
        return self.get_prop('midtransit')

    @property
    def star_mag(self):
        """ """
        return self.get_prop('star_mag')

    def _print(self, msg, *args, **kwargs):
        if self.verbose:
            print(msg, *args, **kwargs)
