import os
import numpy as np
import requests

import gzip
import io

from warnings import warn

from astropy.time import Time
from astropy.time import TimeDelta
from astropy.table import Table

from astropy import units as u
from collections import namedtuple

from dateutil.parser import parse as date_parse

Transit = namedtuple('Transit', ['ingress', 'midpoint', 'egress'])


class Exoplanet():

    def __init__(self, name, planets_file=None, verbose=False, *args, **kwargs):
        self.verbose = verbose
        
        if planets_file is None:
            planets_file = os.path.join(os.environ['PANDIR'], 'PIAA', 'resources', 'planets.csv')
            
        assert os.path.exists(planets_file)

        self.exo_table = Table.read(planets_file, format='ascii.csv', comment='#', header_start=0, data_start=1)
        
        # Make planet names that match ours
        self.exo_table['clean_name'] = [row.title().replace(' ', '').replace('-', '') for row in self.exo_table['pl_hostname']]
        self.exo_table.add_index('clean_name')
        
        self.info = self.exo_table.loc[name]
        
        self.lookups = dict()
        
    @property
    def name(self):
        return self.info['pl_hostname']

    @property
    def transit_midpoint(self):
        """ """
        val = self.info['pl_tranmid'] or self.lookups.get('midpoint', None)
        if not val:
            self._print("Looking up midpoint")
            val = self.lookup_column('mpl_tranmid')
            self.lookups['midpoint'] = val
            
        return Time(val, format='jd')
    
    @property
    def transit_depth(self):
        """ """
        val = self.info['pl_trandep'] or self.lookups.get('depth', None)
        if not val:
            self._print("Looking up depth")
            val = self.lookup_column('mpl_trandep')
            self.lookups['depth'] = val
            
        return val
    
    @property
    def transit_duration(self):
        """ """
        val = self.info['pl_trandur'] or self.lookups.get('duration', None)
        if not val:
            self._print("Looking up duration")
            val = self.lookup_column('mpl_trandur')
            self.lookups['duration'] = val
            
        return TimeDelta(val * u.day)
        
    @property
    def period(self):
        """ """
        val = self.info['pl_orbper'] or self.lookups.get('period')
        if not val:
            self._print("Looking up period")
            val = self.lookup_column('mpl_orbper')
            self.lookups['period'] = val
            
        return TimeDelta(val * u.day)
    
    @property
    def star_mag(self):
        """ """
        return self.info['st_optmag']

    def in_transit(self, t0, with_times=False):
        if isinstance(t0, str):
            t0 = Time(t0)

        midtime = self.transit_midpoint
        period_delta = self.period

        num_periods = int((t0 - midtime).sec // period_delta.sec)
        in_transit = False

        for n in range(num_periods, num_periods + 1):
            midpoint = midtime + (n * period_delta)
            ingress = midpoint - (self.transit_duration / 2)
            egress = midpoint + (self.transit_duration / 2)

            in_transit = t0 >= ingress and t0 <= egress

            if in_transit:
                break

        if with_times:
            return (in_transit, Transit(ingress, midpoint, egress))
        else:
            return in_transit    
    
    def phase_from_time(self, t1):
        if isinstance(t1, str):
            t1  = Time(date_parse(t1), format='datetime')
            
        transittime = self.transit_midpoint
        period = self.period

        num_transits = int((t1 - transittime).sec // period.sec)

        time_since_midpoint = (t1 - transittime) - (num_transits * period)

        phase = (time_since_midpoint / period) - 1.0
        
        if phase > 0.5:
            phase -= 1.0
            
        if phase < -0.5:
            phase += 1.0
            
        return phase
    
    def get_phase(self, t0, verbose=False):
        """
        Args:
            t0 (astropy.time.Time): Time at which to get phase.
        """
        if isinstance(t0, str):
            t0 = Time(date_parse(t0))
            
        midtime = self.transit_midpoint
        period_delta = self.period
        
        time_delta = t0.jd - midtime.jd

        phase = (time_delta % period_delta.jd) / period_delta.jd

        if phase >= 0.5:
            phase -= 1.0

        if verbose:
            print("{:.02f} \t {} \t {:.02f} \t {:.02f} \t {:.02f} \t {:.02f}".format(
                period_delta.jd, 
                t0.isot,
                t0.jd, 
                Time(midtime, format='jd').jd, 
                time_delta,
                phase
            ))

        return phase
    
    def lookup_column(self, col_name):
        info = self.query_jpl(additional_columns=[col_name])
        return [row for row in info if row[col_name] is not None][0][col_name]
    
    def query_jpl(self, out_format='json', additional_columns=None, verbose=False):

        default_columns = ['mpl_hostname','mpl_def','ra','dec','mpl_trandur','mpl_trandep','mpl_tranmid']
        
        if additional_columns:
            if isinstance(additional_columns, str):
                additional_columns = list(additional_columns)
                
            default_columns.extend(additional_columns)
        
        base_url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=multiexopars'
        columns = 'select={}'.format(','.join(default_columns))

        out_format = 'format={}'.format(out_format)
        where_clause = "where=mpl_hostname='{}'".format(self.name)
        
        full_url = '&'.join([base_url, columns, out_format, where_clause])
        
        if verbose:
            print(full_url)
            
        response = requests.get(full_url)
            
        if 'json' in out_format:
            output = response.json()
        else:
            output = response.content
            
        return output
        
    def _print(self, msg, *args, **kwargs):
        if self.verbose:
            print(msg, *args, **kwargs)