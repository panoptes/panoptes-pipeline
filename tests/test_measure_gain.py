"""`scripts/measure_gain.py`: the fits behind the camera registry's gain, black and clip."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import measure_gain  # noqa: E402


def synthetic_blocks(gain: float, black: float, offsets: dict[str, float], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sequence, offset in offsets.items():
        for pair in range(20):
            level = rng.uniform(black + 300, black + 2500, size=200)
            variance = (level - black) / gain + offset
            variance *= 1 + rng.normal(0, 0.03, size=level.size)
            for k, (x, y) in enumerate(zip(level, variance)):
                rows.append((sequence, pair, "G2 (0,0)", k, x, y, 1.0))
    table = pd.DataFrame(
        rows, columns=["sequence", "pair", "channel", "block", "level", "variance", "fraction"]
    )
    table["pair_id"] = table.sequence + ":" + table.pair.astype(str)
    return table


def test_per_sequence_intercepts_recover_the_slope_when_a_single_line_cannot():
    """Two sequences with different variance offsets: one intercept across both turns offset into slope."""
    table = synthetic_blocks(gain=2.3, black=2048, offsets={"a": 40.0, "b": 400.0})
    fit = measure_gain.fit_group(table, black=2048)
    assert abs(fit["gain"] - 2.3) < 0.03
    assert abs(fit["single_line"]["gain"] - 2.3) > 0.03
    assert set(fit["intercepts"]) == {"a", "b"}


def test_fit_line_rejects_a_contaminated_block():
    table = synthetic_blocks(gain=2.0, black=512, offsets={"only": 10.0})
    variance = table.variance.to_numpy().copy()
    variance[:5] *= 20  # a cosmic ray or an unmasked star
    slope, (intercept,), keep = measure_gain.fit_line(table.level.to_numpy(), variance)
    assert abs(1 / slope - 2.0) < 0.02
    assert not keep[:5].any()


def test_pairs_need_a_sky_range_to_be_fitted():
    table = synthetic_blocks(gain=2.3, black=2048, offsets={"a": 40.0})
    flat = table.copy()
    flat["level"] = 2100.0  # no lever arm at all
    drift = pd.Series({pair_id: 0.1 * n for n, pair_id in enumerate(table.pair_id.unique())})
    assert measure_gain.fit_pairs(flat, drift, min_range=150)["num_pairs"] == 0
    result = measure_gain.fit_pairs(table, drift, min_range=150)
    assert result["num_pairs"] == 20
    assert abs(result["gain_median"] - 2.3) < 0.05


def test_find_clip_counts_the_pile_with_its_neighbour():
    histogram = np.zeros(measure_gain.ADC_LEVELS, dtype=np.int64)
    histogram[11000:13583] = 5
    histogram[13583] = 100_000
    histogram[13584] = 40_000  # the same saturation, rounded up on some frames
    histogram[14000] = 3  # a hot pixel
    clip = measure_gain.find_clip(histogram, white_level=11765)
    assert clip["clip"] == 13583
    assert clip["pixels_at_clip"] == 140_005
    assert clip["pixels_above_clip"] == 3


def _sky_histogram(scale: float, black: int = 2048, clip: int = 16383) -> np.ndarray:
    """A sky histogram whose raw values were scaled by `scale` above black."""
    rng = np.random.default_rng(1)
    electrons = rng.normal(900, 60, size=2_000_000).clip(0)
    raw = black + np.rint(np.rint(electrons) * scale).astype(int)
    return np.bincount(np.minimum(raw, clip), minlength=measure_gain.ADC_LEVELS)


def test_find_scale_reads_the_comb_of_a_scaled_raw():
    scaled = measure_gain.find_scale(_sky_histogram(1.61), black=2048, clip=16383)
    assert abs(scaled["scale"] - 1.61) < 0.03
    assert scaled["codes"] > 100

    plain = measure_gain.find_scale(_sky_histogram(1.0), black=2048, clip=13583)
    assert plain["scale"] == 1.0

    empty = measure_gain.find_scale(np.zeros(measure_gain.ADC_LEVELS, dtype=int), black=2048, clip=13583)
    assert np.isnan(empty["scale"])


def test_find_clip_needs_a_pile_not_just_a_maximum():
    rng = np.random.default_rng(2)
    histogram = np.zeros(measure_gain.ADC_LEVELS, dtype=int)
    histogram[11765:13700] = rng.integers(0, 12, size=13700 - 11765)  # hot pixels, no pile
    histogram[13211] = 40  # one unlucky bin
    clip = measure_gain.find_clip(histogram, white_level=11765)
    assert clip["clip"] is None
    assert clip["pixels_at_clip"] == 0

    histogram[13583] = 200_000
    assert measure_gain.find_clip(histogram, white_level=11765)["clip"] == 13583


def test_fit_group_with_no_blocks_reports_nan_not_a_crash():
    empty = pd.DataFrame(columns=["sequence", "pair_id", "level", "variance"])
    fit = measure_gain.fit_group(empty, black=2048)
    assert fit["num_blocks"] == 0 and np.isnan(fit["gain"])
