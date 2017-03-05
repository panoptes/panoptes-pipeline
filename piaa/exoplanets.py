import numpy as np
import urllib

import gzip
import io
import xml.etree.ElementTree as ET

from astropy.time import Time
from astropy.time import TimeDelta

from astropy import units as u
from collections import namedtuple

Transit = namedtuple('Transit', ['ingress', 'midpoint', 'egress'])


class OEC(object):

    def __init__(self, file=None, *args, **kwargs):
        if file is None:
            url = "https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz"
            file = io.BytesIO(urllib.request.urlopen(url).read())
            self.oec = ET.parse(gzip.GzipFile(fileobj=file))
        else:
            self.oec = ET.parse(gzip.GzipFile(filename=file))

        self.names = list()

    def _build_properties(self, element):
        for elem in element:
            if elem.tag != 'name':
                try:
                    setattr(self, elem.tag, float(elem.text))
                except ValueError:
                    setattr(self, elem.tag, elem.text)
                except AttributeError:
                    pass
                except TypeError:
                    pass
            else:
                self.names.append(elem.text)

    @property
    def name(self):
        return self.names[0]


class Star(OEC):

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.system = self.oec.findall(".//system[name='{}']".format(name))[0]
        element = self.system.find('.//star')
        self._build_properties(element)

        self.planet = Exoplanet(self.system.find('.//planet'), star=self)


class Exoplanet(OEC):

    def __init__(self, element, star=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_properties(element)
        self.star = star

        self._transit_duration = None
        self._b = None
        self._transit_depth = None

    @property
    def transit_duration(self):
        """ Get transit duration in minutes

        Seager, S., & Mallen-Ornelas, G. (2002).
        The Astrophysical Journal, 585, 1038–1055. https://doi.org/10.1086/346105
        """
        if self._transit_duration is None:
            i = self.inclination * u.degree
            r_s = self.star.radius * u.R_sun
            r_p = self.radius * u.R_jup
            period = self.period * u.day
            a = self.semimajoraxis * u.AU

            self._transit_duration = ((period / np.pi) * np.arcsin(
                (r_s / a) *
                np.sqrt(
                    (
                        (1 + (r_p / r_s))**2 -
                        ((a / r_s) * np.cos(i))**2
                    ) / (1 - np.cos(i)**2)
                )
            ).value).to(u.minute).value

        return self._transit_duration

    @property
    def b(self):
        if self._b is None:
            a = (self.semimajoraxis * u.AU).to(u.m)
            r_s = (self.star.radius * u.R_sun).to(u.m)
            i = self.inclination * u.degree
            self._b = (a / r_s) * np.cos(i)

        return self._b

    @property
    def impact_parameter(self):
        return self.b

    @property
    def transit_depth(self):
        if self._transit_depth is None:
            r_s = (self.star.radius * u.R_sun).to(u.m)
            r_p = (self.radius * u.R_jup).to(u.m)
            self._transit_depth = (r_p / r_s)**2

        return self._transit_depth

    def in_transit(self, t1, with_times=False):
        if isinstance(t1, str):
            t1 = Time(t1)

        transittime = Time(self.star.planet.transittime, format='jd')
        period_delta = TimeDelta(self.star.planet.period, format='jd')

        num = int((t1 - transittime).sec // period_delta.sec)
        in_transit = False

        for n in range(num, num + 2):
            midpoint = transittime + (n * period_delta)
            ingress = midpoint - (self.transit_duration * u.minute / 2)
            egress = midpoint + (self.transit_duration * u.minute / 2)

            in_transit = t1 >= ingress and t1 <= egress

            if in_transit:
                break

        if with_times:
            return (in_transit, Transit(ingress.isot, midpoint.isot, egress.isot))
        else:
            return in_transit

    def transits_in_range(self, t0, t1, num_of_transits=10):
        if isinstance(t0, str):
            t0 = Time(t0)

        if isinstance(t1, str):
            t1 = Time(t1)

        transittime = Time(self.star.planet.transittime, format='jd')
        period_delta = TimeDelta(self.star.planet.period, format='jd')

        # Get nearest mid-transit point
        num = int((t0 - transittime).sec // period_delta.sec) - 1

        transits = list()

        while True:
            midpoint = transittime + (num * period_delta)
            ingress = midpoint - (self.transit_duration * u.minute / 2)
            egress = midpoint + (self.transit_duration * u.minute / 2)

            if ingress > t1:
                break

            transits.append(Transit(ingress.isot, midpoint.isot, egress.isot))

            num += 1

        return transits
