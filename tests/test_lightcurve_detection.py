"""Tests for the detection statistics -- the objective, per algorithm design 4."""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.pipeline.lightcurve import detection, injection


def series(num=400, cadence_s=35.0, scatter=0.005, seed=0):
    times = np.arange(num, dtype=float) * cadence_s
    return times, 1.0 + np.random.default_rng(seed).normal(0, scatter, num)


def test_depth_is_recovered_with_a_free_baseline():
    times, flux = series(scatter=0.0)
    flux = flux * injection.box_transit(times, times[200], 1.0, 0.02)
    weights = detection.transit_weights(times, times[200], 1.0, ingress_fraction=0.0)
    depth, _ = detection.fit_depth(flux, weights)
    assert depth == pytest.approx(0.02, abs=1e-6)


def test_fractional_depth_is_invariant_under_rescaling():
    """Required for stitching: two units on different scales must agree on depth.

    Lightcurves are no longer normalised to unit baseline (algorithm design 6),
    so the depth has to be measured as a fraction of the fitted baseline or
    segments from different units could never be compared.
    """
    times, flux = series(scatter=0.0)
    flux = flux * injection.box_transit(times, times[200], 1.0, 0.02)
    weights = detection.transit_weights(times, times[200], 1.0, ingress_fraction=0.0)

    for scale in (0.5, 1.0, 3.7, 480.0):
        depth, _ = detection.fit_depth(flux * scale, weights)
        assert depth == pytest.approx(0.02, rel=1e-6)


def test_an_additive_pedestal_dilutes_the_depth():
    """And it should -- that is what an un-subtracted background does.

    A multiplicative scale leaves the fraction alone; an additive offset does
    not, which is exactly the dilution measured in conformance audit 5.0.
    """
    times, flux = series(scatter=0.0)
    flux = flux * injection.box_transit(times, times[200], 1.0, 0.02)
    weights = detection.transit_weights(times, times[200], 1.0, ingress_fraction=0.0)

    pedestal = 0.037
    depth, _ = detection.fit_depth(flux + pedestal, weights)
    assert depth == pytest.approx(0.02 / (1.0 + pedestal), rel=1e-6)
    assert depth < 0.02


def test_snr_grows_with_depth_and_is_near_zero_without_a_signal():
    times, noise = series(seed=1)
    quiet = detection.transit_snr(times, noise, times[200], 1.0)
    assert abs(quiet) < 4.0

    previous = 0.0
    for depth in (0.005, 0.01, 0.02):
        flux = noise * injection.box_transit(times, times[200], 1.0, depth)
        snr = detection.transit_snr(times, flux, times[200], 1.0)
        assert snr > previous
        previous = snr


def test_blind_scan_finds_an_injected_transit():
    times, noise = series(scatter=0.002, seed=2)
    flux = noise * injection.trapezoid_transit(times, times[250], 2.0, 0.02)
    found = detection.scan(times, flux, durations_hours=(1.0, 2.0, 4.0))
    assert found.snr > 8.0
    assert found.depth == pytest.approx(0.02, rel=0.4)
    assert abs(found.mid_transit - times[250]) < 2.0 * 3600.0


def test_scan_on_pure_noise_stays_modest():
    times, noise = series(seed=3)
    assert detection.scan(times, noise).snr < 6.0


def test_red_noise_raises_the_false_alarm_threshold():
    """The point of the metric: correlated noise makes real detection harder.

    RMS alone cannot see this -- both series are scaled to the same scatter.
    """
    times, white = series(num=600, scatter=0.004, seed=4)
    red = 1.0 + 0.004 * np.sin(2 * np.pi * times / (2.5 * 3600.0))
    red = 1.0 + (red - 1.0) * (np.std(white - 1.0) / np.std(red - 1.0))

    white_threshold = detection.false_alarm_threshold(times, white, num_trials=60)
    red_threshold = detection.false_alarm_threshold(times, red, num_trials=60)
    assert red_threshold > white_threshold


def test_circular_shift_null_is_not_a_shuffle():
    """A shuffled null would erase the correlation the threshold must capture."""
    times, _ = series(num=600)
    red = 1.0 + 0.01 * np.sin(2 * np.pi * times / (2.0 * 3600.0))
    shifted = detection.false_alarm_threshold(times, red, num_trials=60, seed=5)
    shuffled = detection.false_alarm_threshold(
        times, np.random.default_rng(0).permutation(red), num_trials=60, seed=5
    )
    assert shifted > shuffled


def test_completeness_rises_with_depth():
    times, noise = series(num=500, scatter=0.004, seed=6)

    def recover(depth, duration_hours):
        rng = np.random.default_rng()
        centre = rng.uniform(times[100], times[-100])
        return noise * injection.trapezoid_transit(times, centre, duration_hours, depth)

    grid = detection.completeness(
        recover, times, depths=(0.002, 0.03), durations_hours=(2.0,),
        threshold=7.0, num_phases=6,
    )
    assert grid[(0.002, 2.0)] < grid[(0.03, 2.0)]
    assert grid[(0.03, 2.0)] > 0.5


def test_completeness_keys_cover_the_grid():
    times, noise = series(num=200)
    grid = detection.completeness(
        lambda d, t: noise, times, depths=(0.01, 0.02),
        durations_hours=(1.0, 2.0), threshold=1e9, num_phases=1,
    )
    assert set(grid) == {(0.01, 1.0), (0.01, 2.0), (0.02, 1.0), (0.02, 2.0)}
    assert all(value == 0.0 for value in grid.values())
