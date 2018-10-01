import pytest

from piaa.exoplanets import Exoplanet


@pytest.fixture(scope='function')
def exoplanet():
    return Exoplanet('HD 189733 b')


def test_exists(exoplanet):
    assert exoplanet is not None


def test_no_exists():
    with pytest.raises(Exception):
        Exoplanet('Foo Bar b')


def test_get_prop(exoplanet):
    assert exoplanet.get_prop('star_mag')
    assert exoplanet.get_prop('foobar') is None
    assert exoplanet.transit_duration == exoplanet.get_prop('transit_duration')
    assert exoplanet.period == exoplanet.get_prop('period')
    assert exoplanet.midtransit.value == exoplanet.get_prop('midtransit')
    assert exoplanet.star_mag == exoplanet.get_prop('star_mag')

# # Calculate next transit times which occur after first image
# obs_time = Time(image_times[0])

# next_transit = hd189733.next_primary_eclipse_time(obs_time)

# # Get the ingress and egress times for the transit
# ing_egr = hd189733.next_primary_ingress_egress_time(obs_time)

# # Get transit properties
# midpoint = next_transit[0].datetime
# ingress = ing_egr.datetime[0][0]
# egress = ing_egr.datetime[0][1]
