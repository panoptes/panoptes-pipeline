"""Tests for the precision metrics used to score every algorithm change."""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.pipeline.lightcurve import metrics


def white_noise(num_points=2000, scatter=0.01, seed=0):
    times = np.arange(num_points, dtype=float) * 60.0
    flux = 1.0 + np.random.default_rng(seed).normal(0.0, scatter, num_points)
    return times, flux


def test_rms_measures_fractional_scatter():
    _, flux = white_noise(scatter=0.02, seed=1)
    assert metrics.rms(flux) == pytest.approx(0.02, rel=0.1)


def test_binning_white_noise_follows_root_n():
    times, flux = white_noise(num_points=3600, scatter=0.01, seed=2)
    unbinned = metrics.rms(flux)
    _, binned, counts = metrics.bin_flux(times, flux, bin_minutes=30)
    expected = unbinned / np.sqrt(counts.mean())
    assert metrics.rms(binned) == pytest.approx(expected, rel=0.25)


def test_zero_bin_width_returns_the_series_unchanged():
    times, flux = white_noise(num_points=50)
    out_times, out_flux, counts = metrics.bin_flux(times, flux, bin_minutes=0)
    assert np.array_equal(out_flux, flux)
    assert counts.sum() == len(flux)


def test_beta_is_about_one_for_white_noise():
    times, flux = white_noise(num_points=3600, seed=3)
    assert metrics.beta_factor(times, flux, bin_minutes=30) == pytest.approx(1.0, abs=0.3)


def test_beta_exceeds_one_when_noise_is_correlated():
    times, white = white_noise(num_points=2000, scatter=0.005, seed=4)
    red = white + 0.01 * np.sin(2 * np.pi * times / (90 * 60.0))
    assert metrics.beta_factor(times, red, bin_minutes=30) > metrics.beta_factor(
        times, white, bin_minutes=30
    )


def test_precision_curve_improves_with_bin_size():
    times, flux = white_noise(num_points=3600, seed=5)
    curve = metrics.precision_curve(times, flux, bin_minutes=(0, 10, 30))
    assert curve[30.0] < curve[10.0] < curve[0.0]


def test_photon_noise_floor_scales_as_root_n():
    bright = metrics.photon_noise_floor(1e6, gain_e_per_adu=1.5, read_noise_e=0.0)
    faint = metrics.photon_noise_floor(1e4, gain_e_per_adu=1.5, read_noise_e=0.0)
    assert faint / bright == pytest.approx(10.0, rel=0.01)


def test_photon_noise_floor_refuses_to_guess_camera_constants():
    """A fleet-wide default gain would give a wrong floor for every camera but one."""
    with pytest.raises(TypeError):
        metrics.photon_noise_floor(1e6)


def test_photon_noise_floor_tracks_gain():
    low = metrics.photon_noise_floor(1e5, gain_e_per_adu=0.5, read_noise_e=0.0)
    high = metrics.photon_noise_floor(1e5, gain_e_per_adu=2.0, read_noise_e=0.0)
    assert low / high == pytest.approx(2.0, rel=0.01)


def test_report_ratios_against_the_noise_floor():
    times, flux = white_noise(num_points=200, scatter=0.02, seed=6)
    card = metrics.report(times, flux, num_active_references=46, noise_floor=0.01)
    assert card.floor_ratio == pytest.approx(2.0, rel=0.15)
    assert card.num_frames == 200
    assert "unbinned_rms" in card.to_dict()


def test_datetime_times_are_accepted():
    stamps = np.datetime64("2026-01-01T00:00:00") + np.arange(120) * np.timedelta64(35, "s")
    flux = np.ones(120)
    _, binned, _ = metrics.bin_flux(stamps, flux, bin_minutes=10)
    assert len(binned) >= 6
