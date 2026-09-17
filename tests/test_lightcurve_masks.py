"""Tests for Bayer masks and apertures."""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.pipeline.lightcurve import masks


def test_rggb_superpixel_has_one_red_two_green_one_blue():
    rgb = masks.rgb_masks((10, 10))
    assert rgb["r"].sum() == 25
    assert rgb["g"].sum() == 50
    assert rgb["b"].sum() == 25
    assert (rgb["r"] | rgb["g"] | rgb["b"]).all()


def test_masks_are_mutually_exclusive():
    rgb = masks.rgb_masks((8, 8))
    assert not (rgb["r"] & rgb["g"]).any()
    assert not (rgb["g"] & rgb["b"]).any()
    assert not (rgb["r"] & rgb["b"]).any()


def test_odd_origin_shifts_the_bayer_phase():
    """A stamp that is not superpixel-aligned carries a different color layout."""
    aligned = masks.rgb_masks((4, 4), origin=(0, 0))
    shifted = masks.rgb_masks((4, 4), origin=(1, 1))
    assert not np.array_equal(aligned["r"], shifted["r"])
    assert np.array_equal(aligned["r"], shifted["b"])


def test_every_supported_pattern_round_trips():
    for pattern in masks.PATTERNS:
        rgb = masks.rgb_masks((6, 6), pattern=pattern)
        assert rgb["g"].sum() == 18
        assert rgb["r"].sum() == rgb["b"].sum() == 9


def test_unknown_pattern_is_rejected():
    with pytest.raises(ValueError, match="Unknown Bayer"):
        masks.bayer_color_index((4, 4), pattern="XYZW")


def test_superpixel_alignment_check():
    assert masks.is_superpixel_aligned((100, 250), (10, 10))
    assert not masks.is_superpixel_aligned((101, 250), (10, 10))
    assert not masks.is_superpixel_aligned((100, 250), (9, 10))


def test_grow_to_superpixels_preserves_the_color_ratio():
    aperture = masks.circular_aperture((10, 10), center=(4.5, 4.5), radius=2.4)
    grown = masks.grow_to_superpixels(aperture)
    rgb = masks.rgb_masks((10, 10))

    assert grown.sum() >= aperture.sum()
    assert (rgb["g"] & grown).sum() == 2 * (rgb["r"] & grown).sum()
    assert (rgb["r"] & grown).sum() == (rgb["b"] & grown).sum()


# Sub-pixel centroids a star actually drifts through during a sequence.
CENTROIDS = [(4.0, 4.0), (4.0, 5.0), (4.5, 4.5), (5.0, 4.0), (5.0, 5.0)]


def test_raw_aperture_color_ratio_swings_with_sub_pixel_position():
    """The motivation for grow_to_superpixels.

    A hard-edged aperture clips whole superpixels differently depending on where
    the star sits, so the green:red pixel ratio inside it is not fixed at 2:1.
    Across half-pixel centroid shifts the ratio swings by more than a factor of
    four, which is a color-dependent flux error the algorithm never sees. See
    conformance audit 3.5 and improvement plan 3.6.
    """
    rgb = masks.rgb_masks((10, 10))
    ratios = []
    for center in CENTROIDS:
        aperture = masks.circular_aperture((10, 10), center=center, radius=2.0)
        ratios.append((rgb["g"] & aperture).sum() / max((rgb["r"] & aperture).sum(), 1))

    assert max(ratios) / min(ratios) > 4.0
    assert not all(ratio == 2.0 for ratio in ratios)


def test_growing_to_superpixels_fixes_the_ratio_at_every_centroid():
    """The fix: whatever the sub-pixel position, the aperture stays 1:2:1."""
    rgb = masks.rgb_masks((10, 10))
    for center in CENTROIDS:
        aperture = masks.circular_aperture((10, 10), center=center, radius=2.0)
        grown = masks.grow_to_superpixels(aperture)
        num_red = (rgb["r"] & grown).sum()
        assert (rgb["g"] & grown).sum() == 2 * num_red
        assert (rgb["b"] & grown).sum() == num_red


def test_weighted_aperture_recovers_total_flux():
    profile = np.array([0.1, 0.4, 0.4, 0.1])
    weights = masks.weighted_aperture(profile, noise_variance=1.0)
    assert float((weights * profile).sum()) == pytest.approx(1.0)


def test_weighted_aperture_rejects_a_null_profile():
    with pytest.raises(ValueError, match="Degenerate"):
        masks.weighted_aperture(np.zeros(4), noise_variance=1.0)


# --- Inferring the pattern from data rather than assuming it ---------------


def _synthetic_sky(stamp_shape, pattern, levels=(700.0, 770.0, 640.0), seed=0):
    """Star-free stamps with a per-color sky level, in a known pattern."""
    rng = np.random.default_rng(seed)
    colors = masks.bayer_color_index(stamp_shape, pattern=pattern)
    sky = np.zeros(stamp_shape)
    for value, level in zip((masks.RED, masks.GREEN, masks.BLUE), levels):
        sky[colors == value] = level
    return sky + rng.normal(0, 2.0, (40, *stamp_shape))


@pytest.mark.parametrize("pattern", sorted(masks.PATTERNS))
def test_green_offsets_recovered_for_every_pattern(pattern):
    stamps = _synthetic_sky((10, 18), pattern)
    greens = masks.infer_green_offsets(stamps.reshape(40, -1), (10, 18))
    expected = {(dy, dx) for dy in (0, 1) for dx in (0, 1) if masks.PATTERNS[pattern][dy][dx] == masks.GREEN}
    assert greens == expected


@pytest.mark.parametrize("pattern", sorted(masks.PATTERNS))
def test_pattern_round_trips_given_the_red_offset(pattern):
    stamps = _synthetic_sky((10, 18), pattern)
    red = next((dy, dx) for dy in (0, 1) for dx in (0, 1) if masks.PATTERNS[pattern][dy][dx] == masks.RED)
    assert masks.infer_pattern(stamps.reshape(40, -1), (10, 18), red_offset=red) == pattern


def test_infer_pattern_refuses_to_guess_red_versus_blue():
    """Guessing silently swaps the red and blue lightcurves."""
    stamps = _synthetic_sky((10, 18), "RGGB")
    with pytest.raises(ValueError, match="cannot be read from sky levels"):
        masks.infer_pattern(stamps.reshape(40, -1), (10, 18))


def test_infer_pattern_rejects_a_green_position_as_red():
    stamps = _synthetic_sky((10, 18), "RGGB")
    with pytest.raises(ValueError, match="is a green position"):
        masks.infer_pattern(stamps.reshape(40, -1), (10, 18), red_offset=(0, 1))
