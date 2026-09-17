"""Tests for the paper-faithful algorithm core (paper sections 3.2.1 - 3.2.5)."""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.pipeline.lightcurve import core, injection, masks, metrics
from tests.synthetic import make_observation

# --- 3.2.1 Prepare PSCs ---------------------------------------------------


def test_normalize_psc_frames_sum_to_one():
    psc = np.arange(1, 21, dtype=float).reshape(4, 5)
    normed = core.normalize_psc(psc)
    assert np.allclose(normed.sum(axis=1), 1.0)


def test_normalize_psc_handles_stacked_cubes():
    cube = np.random.default_rng(0).uniform(1, 10, size=(3, 4, 5))
    normed = core.normalize_psc(cube)
    assert normed.shape == cube.shape
    assert np.allclose(normed.sum(axis=2), 1.0)


def test_normalize_psc_marks_dead_frames_nan_not_zero():
    psc = np.array([[1.0, 1.0], [0.0, 0.0]])
    normed = core.normalize_psc(psc)
    assert np.allclose(normed[0], 0.5)
    assert np.isnan(normed[1]).all(), "A zero-flux frame must not silently become zeros"


def test_normalize_psc_pixel_mask_restricts_the_sum():
    psc = np.array([[1.0, 3.0, 4.0, 2.0]])
    mask = np.array([True, True, False, False])
    normed = core.normalize_psc(psc, pixel_mask=mask)
    assert np.allclose(normed[0], [0.25, 0.75, 0.0, 0.0])


# --- 3.2.2 Find reference stars -------------------------------------------


def test_identical_star_scores_zero():
    obs = make_observation(num_stars=5, num_frames=6, seed=1)
    normed = core.normalize_psc(obs.pscs)
    scores = core.similarity_scores(normed[0], normed)
    assert scores[0] == pytest.approx(0.0, abs=1e-12)
    assert (scores[1:] > 0).all()


def test_select_references_refuses_a_pool_containing_the_target():
    with pytest.raises(ValueError, match="target is present"):
        core.select_references(np.array([0.0, 1.0, 2.0]))


def test_select_references_returns_best_first():
    chosen = core.select_references(np.array([5.0, 1.0, 3.0, 2.0]), num_refs=3)
    assert chosen.tolist() == [1, 3, 2]


def test_frame_weights_exclude_frames_from_scoring():
    obs = make_observation(num_stars=4, num_frames=6, seed=2)
    normed = core.normalize_psc(obs.pscs)
    weights = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    weighted = core.similarity_scores(normed[0], normed[1:], frame_weights=weights)
    full = core.similarity_scores(normed[0], normed[1:])
    assert (weighted < full).all()


# --- 3.2.3 Determine coefficients -----------------------------------------


def test_ols_recovers_an_exact_linear_combination():
    """If the target *is* a mixture of the references, the fit must find it."""
    rng = np.random.default_rng(7)
    refs = rng.uniform(0.5, 1.5, size=(6, 8, 12))
    refs = core.normalize_psc(refs)
    truth = np.array([0.4, -0.15, 0.25, 0.0, 0.3, 0.2])
    target = np.tensordot(truth, refs, axes=(0, 0))

    coeffs = core.solve_coefficients(target, refs, method="ols")
    assert np.allclose(coeffs, truth, atol=1e-8)


def test_lasso_drives_coefficients_to_exactly_zero():
    """Paper Fig. 7 shows 46 of 100 coefficients at zero - an L1 signature."""
    obs = make_observation(num_stars=60, num_frames=12, seed=3)
    normed = core.normalize_psc(obs.pscs)

    ols = core.solve_coefficients(normed[0], normed[1:], method="ols")
    lasso = core.solve_coefficients(normed[0], normed[1:], method="lasso", alpha=1e-6)

    assert np.count_nonzero(ols) == len(ols)
    assert 0 < np.count_nonzero(lasso) < len(lasso)


def test_nnls_never_returns_a_negative_coefficient():
    obs = make_observation(num_stars=30, num_frames=10, seed=4)
    normed = core.normalize_psc(obs.pscs)
    coeffs = core.solve_coefficients(normed[0], normed[1:], method="nnls")
    assert (coeffs >= 0).all()


def test_unknown_solver_is_rejected():
    obs = make_observation(num_stars=5, num_frames=4, seed=5)
    normed = core.normalize_psc(obs.pscs)
    with pytest.raises(ValueError, match="Unknown solve"):
        core.solve_coefficients(normed[0], normed[1:], method="magic")


# --- 3.2.4 / 3.2.5 Build comparison and do photometry ---------------------


def test_build_comparison_checks_coefficient_count():
    refs = np.ones((3, 4, 5))
    with pytest.raises(ValueError, match="coefficients for"):
        core.build_comparison(refs, np.ones(2))


def test_comparison_built_from_raw_flux_gives_a_flat_lightcurve():
    """Coefficients fit in normalized space, applied to flux-carrying stamps."""
    rng = np.random.default_rng(11)
    refs = core.normalize_psc(rng.uniform(0.5, 1.5, size=(5, 10, 9)))
    truth = np.array([0.3, 0.2, 0.1, 0.25, 0.15])
    target = np.tensordot(truth, refs, axes=(0, 0))

    comparison = core.build_comparison(refs, truth)
    flux = core.differential_lightcurve(target, comparison)
    assert np.allclose(flux, 1.0, atol=1e-10)


def test_differential_lightcurve_respects_the_aperture():
    target = np.array([[10.0, 10.0, 1000.0]])
    comparison = np.array([[5.0, 5.0, 1.0]])
    mask = np.array([True, True, False])
    flux = core.differential_lightcurve(target, comparison, pixel_mask=mask, normalize=False)
    assert flux[0] == pytest.approx(2.0)


# --- End to end -----------------------------------------------------------


def test_algorithm_beats_raw_aperture_photometry():
    """The whole point: differential photometry must beat summing the stamp."""
    obs = make_observation(num_stars=120, num_frames=30, seed=13)
    target, pool = obs.pscs[0], obs.pscs[1:]

    result = core.make_lightcurve(target, pool, num_refs=60, method="lasso", alpha=1e-7)

    raw = target.sum(axis=1)
    raw = raw / np.median(raw)

    assert metrics.rms(result.flux) < metrics.rms(raw)
    assert result.flux.shape == (30,)
    assert result.num_active_references > 0


def test_per_channel_lightcurves_are_produced_independently():
    obs = make_observation(num_stars=80, num_frames=20, seed=17)
    rgb = masks.rgb_masks(obs.stamp_shape)

    curves = {
        color: core.make_lightcurve(
            obs.pscs[0],
            obs.pscs[1:],
            num_refs=40,
            channel_mask=rgb[color].ravel(),
            channel=color,
        )
        for color in "rgb"
    }

    for color, result in curves.items():
        assert result.channel == color
        assert np.isfinite(result.flux).all()
    assert not np.allclose(curves["r"].flux, curves["b"].flux)


def test_result_records_its_own_settings():
    obs = make_observation(num_stars=30, num_frames=8, seed=19)
    result = core.make_lightcurve(obs.pscs[0], obs.pscs[1:], num_refs=10, method="ridge")
    assert result.meta["method"] == "ridge"
    assert result.meta["num_refs"] == 10
    assert result.meta["pool_size"] == 29
    assert len(result.reference_indices) == 10


# --- Injection and recovery ----------------------------------------------


def test_injected_transit_is_recovered_without_large_suppression():
    obs = make_observation(num_stars=140, num_frames=60, seed=23)
    model = injection.trapezoid_transit(
        obs.times, mid_transit=obs.times[len(obs.times) // 2], duration_hours=0.35, depth=0.02
    )
    target = injection.inject(obs.pscs[0], model)

    result = core.make_lightcurve(target, obs.pscs[1:], num_refs=60, method="lasso", alpha=1e-7)
    recovery = injection.measure_depth(result.flux, model)

    assert recovery.recovered_depth == pytest.approx(0.02, abs=0.006)
    assert abs(recovery.suppression) < 0.3


def test_injection_requires_matching_frame_counts():
    with pytest.raises(ValueError, match="frames"):
        injection.inject(np.ones((5, 4)), np.ones(3))


def test_measure_depth_rejects_a_flat_model():
    with pytest.raises(ValueError, match="no transit"):
        injection.measure_depth(np.ones(10), np.ones(10))


# --- Sky pedestal removal -------------------------------------------------


def test_central_sky_mask_excludes_the_core():
    sky = core.central_sky_mask((10, 18), core_radius=3.0).reshape(10, 18)
    assert not sky[4, 8]
    assert sky[0, 0]
    assert sky.sum() == 180 - 36


def test_subtract_stamp_sky_removes_a_flat_pedestal_per_color():
    """A pedestal 30x the stellar signal must not survive into the photometry."""
    stamp_shape = (10, 18)
    rgb = {name: mask.ravel() for name, mask in masks.rgb_masks(stamp_shape).items()}
    sky = core.central_sky_mask(stamp_shape)

    star = np.zeros((4, 180))
    star[:, np.flatnonzero(~sky)[:8]] = 100.0

    pedestal = np.zeros(180)
    for name, level in zip("rgb", (780.0, 640.0, 900.0)):
        pedestal[rgb[name]] = level

    cleaned = core.subtract_stamp_sky(star + pedestal, sky, rgb)
    assert np.allclose(cleaned, star, atol=1e-9)


def test_subtract_stamp_sky_rejects_an_empty_mask():
    with pytest.raises(ValueError, match="selects no pixels"):
        core.subtract_stamp_sky(np.ones((2, 4)), np.zeros(4, dtype=bool))


def test_pedestal_dilutes_a_transit_and_subtraction_restores_it():
    """Why 5.0 matters: an un-subtracted pedestal shrinks every measured depth."""
    sky = core.central_sky_mask((10, 18))
    star = np.zeros((20, 180))
    star[:, np.flatnonzero(~sky)[:8]] = 100.0
    star[8:12] *= 0.90  # a real 10% transit

    diluted = star + 780.0
    measured = diluted.sum(axis=1) / np.median(diluted.sum(axis=1))
    assert 1.0 - measured.min() < 0.02, "pedestal should crush the apparent depth"

    restored = core.subtract_stamp_sky(diluted, sky)
    recovered = restored.sum(axis=1) / np.median(restored.sum(axis=1))
    assert 1.0 - recovered.min() == pytest.approx(0.10, abs=0.005)


# --- Signal fidelity ------------------------------------------------------


def test_amplitude_is_recovered_regardless_of_phase():
    times = np.arange(400, dtype=float) * 35.0
    for phase in (0.0, 1.1, 2.7, 4.9):
        flux = injection.sinusoid(times, period_hours=2.0, amplitude=0.01, phase=phase)
        assert injection.measure_amplitude(times, flux, 2.0) == pytest.approx(0.01, rel=1e-6)


def test_transfer_is_unity_for_a_pipeline_that_does_nothing():
    times = np.arange(600, dtype=float) * 35.0
    transfer = injection.transfer_function(lambda model: model, times, periods_hours=(0.5, 2.0, 6.0))
    assert all(value == pytest.approx(1.0, abs=0.02) for value in transfer.values())


def test_transfer_exposes_long_timescale_suppression():
    """The failure mode this metric exists for.

    A pipeline that detrends with a polynomial keeps short-period signal and
    eats slow variation. A transit-shaped test at one duration can miss that
    entirely; a sweep over period cannot.
    """
    times = np.arange(800, dtype=float) * 35.0
    scaled = (times - times.mean()) / np.ptp(times)

    def detrending_pipeline(model):
        design = np.vander(scaled, 4)
        coefficients, *_ = np.linalg.lstsq(design, model, rcond=None)
        return model - design @ coefficients + 1.0

    transfer = injection.transfer_function(detrending_pipeline, times, periods_hours=(0.3, 8.0), num_phases=6)
    assert transfer[0.3] > 0.9, "short timescales should pass through"
    assert transfer[8.0] < 0.5, "slow variation should be visibly eaten"


# --- Partial transits and cross-site stitching ----------------------------


def test_a_fully_in_transit_window_keeps_its_depth():
    """The failure mode behind algorithm design 6.

    A long-period transit can exceed one night. Normalizing such a window to
    unit median subtracts the signal from itself and the depth vanishes.
    """
    target = np.full((20, 4), 100.0)
    target[:, :] *= 0.97  # the whole window sits inside the transit
    comparison = np.full((20, 4), 100.0)

    kept = core.differential_lightcurve(target, comparison)
    assert kept.mean() == pytest.approx(0.97, abs=1e-9)

    erased = core.differential_lightcurve(target, comparison, normalize=True)
    assert erased.mean() == pytest.approx(1.0, abs=1e-9)
    assert abs(1.0 - erased.mean()) < 1e-9, "normalizing erased a real 3% dip"


def test_make_lightcurve_preserves_an_absolute_offset():
    """Two windows of the same star must stay on a common scale to be stitched."""
    obs = make_observation(num_stars=60, num_frames=16, seed=31, noise=False)
    dimmed = obs.pscs[0] * 0.98

    bright = core.make_lightcurve(obs.pscs[0], obs.pscs[1:], num_refs=20, method="ols")
    faint = core.make_lightcurve(dimmed, obs.pscs[1:], num_refs=20, method="ols")

    ratio = np.median(faint.flux) / np.median(bright.flux)
    assert ratio == pytest.approx(0.98, rel=0.02)
