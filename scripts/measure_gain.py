#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "astropy",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "scipy",
# ]
# ///
"""Measure a camera's gain, black level and clip from its own science frames.

Improvement plan 1.4, "measure, do not configure", for the three numbers the
camera registry (#135) has to hold: **gain** in e-/ADU, **black level** in raw
ADU, and the raw **clip** where the ADC stops. Nothing here is configured; the
paper's gain of 1.50 is what this exists to check, not an input.

Method
------
Photon transfer on the *sky*, from differences of consecutive frames::

    var(B - A) / 2 = (L - black) / gain + RN**2

where ``L`` is the raw level of the block, black included. The difference of two
frames cancels everything fixed between them -- vignetting, the sky gradient,
PRNU, the dark fixed pattern -- so what is left is the temporal variance alone,
and its slope against level is the gain. The variance is measured per Bayer
channel, because the four channels sit at different sky levels and a slope fitted
across a mixture of them is a slope through four different populations; the
per-channel fits are the measurement, and the pooled fit is reported beside them
so the two can be compared, as #99 asks.

Each block is sigma-clipped and anything the median template calls a star is
masked first. Both matter: an unmasked star adds its own photon noise to the
block, and anything the mask misses adds *more* variance when the pointing
moved between the two frames than when it did not. That second effect is the
systematic this measurement cannot remove -- unresolved stars in a dense field
behave like sky that is a little noisier than sky -- and it biases the gain
**low**. The per-pair gain is therefore also reported against the pointing
drift between the two frames, read from each frame's WCS, with the regression
extrapolated to zero drift. A flat-field photon transfer has no such term, and
is what settles the number; this is what a legacy camera with no flats gets.

The clip is read straight off the raw histogram: the one value above `WHTLVLN`
where the counts pile up. It is the detector's saturation and the registry
stores it as measured, not derived from a header white point.

The same histogram says whether the raw values were scaled before they were
written. A digital scale of the signal above black leaves a comb of empty
codes -- a 1.61 scale populates five codes in eight -- so the populated
fraction of the codes the sky occupies is reported as a scale: 1.00 for an
untouched raw, and the factor to divide the gain by otherwise. PAN019's four
cameras carry a 1.61 with the EXIF still reading ISO 100 (data contract 6.2).

Usage
-----
::

    uv run scripts/measure_gain.py images/PAN012/358d0f/20180824T035917 --output gain/PAN012_358d0f
    uv run scripts/measure_gain.py raw/PAN001/ee04d1/20190527T103457 raw/PAN001/ee04d1/20180913T054325 \\
        --output gain/PAN001_ee04d1

Several sequences from the *same camera and ISO* can be given at once -- they
widen the range of sky levels the line is fitted over, and the fit is better for
it. Mixing cameras is refused. Output is one directory holding the per-block
table, the per-pair table, ``gain.json`` with every fit, and ``gain.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import norm

#: Bayer channels as ``(row offset, column offset)`` into the raw frame. The
#: panoptes-utils convention is superpixel ``[[G2, B], [R, G1]]``; the two
#: diagonal channels are the greens, which the data confirm by their matching
#: sky level. The off-diagonal names are the convention's, not measured.
CHANNELS = {"G2 (0,0)": (0, 0), "B (0,1)": (0, 1), "R (1,0)": (1, 0), "G1 (1,1)": (1, 1)}

#: 14-bit raw: histogram bins cover every value the ADC can produce.
ADC_LEVELS = 2**14


@dataclass
class FrameInfo:
    path: str
    sequence: str
    index: int
    unit: str | None
    camera: str | None
    iso: int | None
    exptime: float | None
    date_obs: str | None
    white_level: int | None
    has_wcs: bool


def read_header(path: Path, sequence: str, index: int) -> FrameInfo:
    from astropy.io import fits

    header = fits.getheader(path, 1) if path.suffix == ".fz" else fits.getheader(path, 0)
    # The unit is the first field of SEQID (``PAN012_358d0f_...``), falling back
    # to the archive path. Not OBSERVER, which is free text ("Mt. Wilson").
    sequence_id = header.get("SEQID") or ""
    unit = sequence_id.split("_")[0] if sequence_id.startswith("PAN") else path.parent.parent.parent.name
    return FrameInfo(
        path=str(path),
        sequence=sequence,
        index=index,
        unit=unit,
        camera=header.get("INSTRUME"),
        iso=header.get("ISO"),
        exptime=header.get("EXPTIME"),
        date_obs=header.get("DATE-OBS"),
        white_level=header.get("WHTLVLN"),
        has_wcs="CRVAL1" in header,
    )


def read_data(path: str) -> np.ndarray | None:
    """The raw frame, or None when the file cannot be decompressed.

    A truncated `.fz` in a ten-year archive is a finding, not a reason to lose
    the run; the caller skips the frame and the pairs it belongs to, and the
    run reports every frame it skipped.
    """
    from astropy.io import fits

    try:
        extension = 1 if str(path).endswith(".fz") else 0
        return np.asarray(fits.getdata(path, extension), dtype=np.uint16)
    except Exception as err:  # noqa: BLE001 -- cfitsio raises its own hierarchy
        print(f"unreadable frame {path}: {err}", file=sys.stderr)
        return None


def blocks(array: np.ndarray, size: int) -> np.ndarray:
    """Reshape a 2-D array into ``(num_blocks, size * size)`` of whole blocks."""
    ny, nx = array.shape
    by, bx = ny // size, nx // size
    whole = array[: by * size, : bx * size].reshape(by, size, bx, size)
    return whole.transpose(0, 2, 1, 3).reshape(by * bx, -1)


def clipped_variance(values: np.ndarray, good: np.ndarray, nsig: float = 4.0, iterations: int = 3):
    """Sigma-clipped variance per row, corrected for the clip's truncation.

    `values` and `good` are ``(num_blocks, num_pixels)``; the variance of a
    Gaussian truncated at ``nsig`` is low by a known factor, which is undone so
    that the clip removes stars and cosmic rays without shaving the noise.
    """
    masked = np.ma.array(values, mask=~good)
    centre = np.asarray(np.ma.median(masked, axis=1))
    scale = np.asarray(np.ma.median(np.abs(masked - centre[:, None]), axis=1)) * 1.4826
    for _ in range(iterations):
        keep = good & (np.abs(values - centre[:, None]) <= nsig * scale[:, None])
        count = keep.sum(axis=1).astype(float)
        centre = (values * keep).sum(axis=1) / np.maximum(count, 1)
        variance = ((values - centre[:, None]) ** 2 * keep).sum(axis=1) / np.maximum(count - 1, 1)
        scale = np.sqrt(variance)
    truncation = 1 - 2 * nsig * norm.pdf(nsig) / (2 * norm.cdf(nsig) - 1)
    return variance / truncation, count / values.shape[1]


def star_mask(template: np.ndarray, threshold: float, dilate: int) -> np.ndarray:
    """Pixels the median template puts above the local sky, per channel, grown by `dilate`."""
    mask = np.zeros(template.shape, dtype=bool)
    for dy, dx in CHANNELS.values():
        plane = template[dy::2, dx::2]
        residual = plane - ndimage.median_filter(plane, size=31)
        sigma = 1.4826 * np.median(np.abs(residual))
        mask[dy::2, dx::2] = ndimage.binary_dilation(residual > threshold * sigma, iterations=dilate)
    return mask


def build_template(paths: list[str], every: int, max_frames: int) -> np.ndarray:
    chosen = paths[::every][:max_frames]
    frames = [f for f in (read_data(p) for p in chosen) if f is not None]
    return np.median(np.stack(frames), axis=0).astype(np.float32)


def pair_variance(args: tuple) -> tuple[list[tuple], np.ndarray]:
    """One frame pair: per-channel block levels and variances, plus the pair's raw histogram."""
    sequence, i, path_a, path_b, mask_path, block, clip_margin, white_level = args
    a = read_data(path_a)
    b = read_data(path_b)
    if a is None or b is None:
        return [], np.zeros(ADC_LEVELS, dtype=np.int64)
    histogram = np.bincount(a.ravel(), minlength=ADC_LEVELS) + np.bincount(b.ravel(), minlength=ADC_LEVELS)
    mask = np.load(mask_path, mmap_mode="r")
    rows = []
    for name, (dy, dx) in CHANNELS.items():
        plane_a = a[dy::2, dx::2].astype(np.float32)
        plane_b = b[dy::2, dx::2].astype(np.float32)
        blocks_a, blocks_b = blocks(plane_a, block), blocks(plane_b, block)
        level = 0.5 * (np.median(blocks_a, axis=1) + np.median(blocks_b, axis=1))
        good = ~blocks(np.asarray(mask[dy::2, dx::2]), block)
        good &= (blocks_a < white_level - clip_margin) & (blocks_b < white_level - clip_margin)
        variance, fraction = clipped_variance(blocks(plane_b - plane_a, block), good)
        for k in range(len(level)):
            rows.append((sequence, i, name, k, float(level[k]), float(variance[k]) / 2.0, float(fraction[k])))
    return rows, histogram


def pointing_offsets(frames: list[FrameInfo]) -> np.ndarray:
    """Pixel shift of the first frame's centre, per frame, from each frame's own WCS. NaN without a WCS."""
    from astropy.io import fits
    from astropy.wcs import WCS

    offsets = np.full((len(frames), 2), np.nan)
    reference = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n, frame in enumerate(frames):
            if not frame.has_wcs:
                continue
            header = fits.getheader(frame.path, 1 if frame.path.endswith(".fz") else 0)
            wcs = WCS(header)
            if reference is None:
                centre = np.array([[header["NAXIS1"] / 2, header["NAXIS2"] / 2]])
                reference = wcs.all_pix2world(centre, 0)
            offsets[n] = wcs.all_world2pix(reference, 0)[0]
    return offsets


def weighted_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    total = w.sum()
    xm, ym = (w * x).sum() / total, (w * y).sum() / total
    slope = (w * (x - xm) * (y - ym)).sum() / (w * (x - xm) ** 2).sum()
    return slope, ym - slope * xm


def fit_line(level: np.ndarray, variance: np.ndarray, groups: np.ndarray | None = None, iterations: int = 4):
    """``variance = slope * level + intercept[group]``, relative-error weights, 4-sigma block rejection.

    One slope, one intercept per group. With `groups` None it is an ordinary
    line. With one group per sequence it is the fit that survives combining
    sequences: every sequence carries its own variance offset -- its field's
    unresolved stars under its night's drift -- and a single intercept across
    sequences turns those offsets into slope.

    A variance estimate's sampling error scales with the variance itself, so
    the weights are ``1 / variance**2`` and the rejection is on the relative
    residual. Returns slope, the intercept per group, and the kept mask.
    """
    weights = 1 / np.maximum(variance, 1) ** 2
    codes = np.zeros(len(level), dtype=int) if groups is None else pd.factorize(groups)[0]
    keep = np.ones(len(level), dtype=bool)
    slope, intercepts = 0.0, np.zeros(codes.max() + 1)
    for _ in range(iterations):
        x_mean, y_mean = np.zeros(len(level)), np.zeros(len(level))
        for code in range(codes.max() + 1):
            member = codes == code
            sel = member & keep
            total = weights[sel].sum()
            x_mean[member] = (weights[sel] * level[sel]).sum() / total
            y_mean[member] = (weights[sel] * variance[sel]).sum() / total
        dx, dy = (level - x_mean)[keep], (variance - y_mean)[keep]
        slope = (weights[keep] * dx * dy).sum() / (weights[keep] * dx * dx).sum()
        for code in range(codes.max() + 1):
            member = codes == code
            intercepts[code] = (y_mean[member] - slope * x_mean[member])[0]
        relative = (variance - y_mean - slope * (level - x_mean)) / np.maximum(variance, 1)
        sigma = 1.4826 * np.median(np.abs(relative[keep]))
        keep = np.abs(relative) < 4 * sigma
    return slope, intercepts, keep


def fit_group(table: pd.DataFrame, black: float, seed: int = 1) -> dict:
    """The fits for one set of blocks: shared slope with a per-sequence intercept, and the plain line.

    The bootstrap resamples *pairs*, not blocks: blocks of one pair share the
    same two frames, so they are not independent draws and resampling them
    would understate the error by the square root of the blocks per pair.
    """
    if len(table) == 0:
        nan = float("nan")
        return {
            "num_blocks": 0,
            "num_rejected": 0,
            "level_min": nan,
            "level_max": nan,
            "gain": nan,
            "gain_err": nan,
            "intercepts": {},
            "black_intercept": {},
            "read_noise_adu": {},
            "read_noise_e": {},
            "single_line": {"gain": nan, "black_intercept": nan, "num_blocks": 0},
        }
    level, variance = table.level.to_numpy(), table.variance.to_numpy()
    sequences = table.sequence.to_numpy()
    slope, intercepts, keep = fit_line(level, variance, sequences)
    slope_single, (intercept_single,), keep_single = fit_line(level, variance)
    by_pair = [idx[keep[idx]] for idx in table.groupby("pair_id").indices.values()]
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(200):
        idx = np.concatenate([by_pair[k] for k in rng.integers(len(by_pair), size=len(by_pair))])
        s, _, _ = fit_line(level[idx], variance[idx], sequences[idx], iterations=1)
        gains.append(1 / s)
    names = list(pd.unique(sequences))
    read_noise_sq = {name: float(slope * black + b) for name, b in zip(names, intercepts)}
    read_noise = {name: float(np.sqrt(max(v, 0.0))) for name, v in read_noise_sq.items()}
    return {
        "num_blocks": int(keep.sum()),
        "num_rejected": int((~keep).sum()),
        "level_min": float(level[keep].min()),
        "level_max": float(level[keep].max()),
        "gain": 1 / slope,
        "gain_err": float(np.std(gains)),
        "intercepts": {name: float(b) for name, b in zip(names, intercepts)},
        "black_intercept": {name: float(-b / slope) for name, b in zip(names, intercepts)},
        "read_noise_adu": read_noise,
        "read_noise_e": {name: float(v / slope) for name, v in read_noise.items()},
        "single_line": {
            "gain": 1 / slope_single,
            "black_intercept": -intercept_single / slope_single,
            "num_blocks": int(keep_single.sum()),
        },
    }


def fit_pairs(table: pd.DataFrame, drift: pd.Series, min_range: float) -> dict:
    """A free line per pair, for every pair with enough sky range across the frame to carry one.

    The per-pair gains are the diagnostic that says whether a sequence can be
    used at all. Anything that adds variance between two frames -- cloud,
    moonlit haze, a bad pair -- lowers that pair's gain, so a clean sequence
    gives a tight distribution and a cloudy one a wide scatter towards low
    values; the spread is reported per sequence, and the drift regression and
    its extrapolation to zero drift use the pairs that pass the range cut.
    """
    rows = []
    for (sequence, pair_id), group in table.groupby(["sequence", "pair_id"]):
        lo, hi = group.level.quantile([0.02, 0.98])
        if hi - lo < min_range or len(group) < 50:
            continue
        slope, (intercept,), _ = fit_line(group.level.to_numpy(), group.variance.to_numpy(), iterations=2)
        d = drift.get(pair_id, np.nan)
        rows.append((sequence, pair_id, float(hi - lo), float(d), 1 / slope, -intercept / slope))
    numeric = ["level_range", "drift", "gain", "black_intercept"]
    pairs = pd.DataFrame(rows, columns=["sequence", "pair_id", *numeric]).astype(
        dict.fromkeys(numeric, float)
    )
    out: dict = {"pairs": pairs, "num_pairs": int(len(pairs))}
    if len(pairs) == 0:
        return out
    out["gain_median"] = float(pairs.gain.median())
    out["gain_p10"] = float(pairs.gain.quantile(0.1))
    out["gain_p90"] = float(pairs.gain.quantile(0.9))
    out["per_sequence"] = {
        name: {
            "num_pairs": int(len(g)),
            "gain_median": float(g.gain.median()),
            "gain_p10": float(g.gain.quantile(0.1)),
            "gain_p90": float(g.gain.quantile(0.9)),
            "median_drift": float(g.drift.median()),
        }
        for name, g in pairs.groupby("sequence")
    }
    with_drift = pairs[np.isfinite(pairs.drift)]
    if len(with_drift) >= 5:
        design = np.vstack([with_drift.drift, np.ones(len(with_drift))]).T
        coef = np.linalg.lstsq(design, with_drift.gain, rcond=None)[0]
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(500):
            s = rng.choice(len(with_drift), len(with_drift))
            boot.append(np.linalg.lstsq(design[s], with_drift.gain.to_numpy()[s], rcond=None)[0][1])
        out.update(
            gain_zero_drift=float(coef[1]),
            gain_zero_drift_err=float(np.std(boot)),
            gain_per_pixel_drift=float(coef[0]),
            correlation=float(np.corrcoef(with_drift.drift, with_drift.gain)[0, 1]),
            median_drift=float(with_drift.drift.median()),
        )
    return out


def find_clip(histogram: np.ndarray, white_level: int, min_pile: int = 100, contrast: float = 20.0) -> dict:
    """The pile-up above the header white level, where every saturated pixel lands.

    The pile is counted with its two neighbours: on some frames the saturated
    value comes out one ADU higher than on others -- 13584 beside 13583 on the
    same camera -- which is a rounding of the raw conversion, not headroom.

    A maximum is not a pile. The tallest bin has to hold at least `min_pile`
    pixels and `contrast` times the median bin above the white level, or no
    frame saturated and the clip is None rather than a hot pixel's value.
    """
    above = histogram[white_level:]
    clip = int(white_level + np.argmax(above))
    pile = int(histogram[clip - 1 : clip + 2].sum())
    floor = float(np.median(above))
    if pile < min_pile or pile < contrast * max(floor, 1.0):
        return {"clip": None, "pixels_at_clip": 0, "pixels_above_clip": 0, "white_level": int(white_level)}
    return {
        "clip": clip,
        "pixels_at_clip": pile,
        "pixels_above_clip": int(histogram[clip + 2 :].sum()),
        "white_level": int(white_level),
    }


def find_scale(histogram: np.ndarray, black: float, clip: int, smooth: int = 15) -> dict:
    """The digital scale of the raw values, from the comb they leave in the histogram.

    A raw scaled by s above black populates one code in s: against a running
    mean over `smooth` codes, a code holding under 30% of its neighbourhood is
    empty. Give it one Bayer channel: each channel's empty codes sit at different
    positions, so a whole frame's comb is partly filled in and reads low.
    The populated fraction is measured where the sky puts the counts --
    between black and the 95th percentile of all pixels, below the clip -- and
    only where the neighbourhood is well sampled, so a sparse tail cannot read
    as a comb. An untouched raw returns 1.00.
    """
    counts = histogram[: clip - 2].astype(float)
    total = counts.sum()
    if total == 0:
        return {"scale": float("nan"), "populated_fraction": float("nan"), "codes": 0}
    p95 = int(np.searchsorted(np.cumsum(counts), 0.95 * total))
    lo, hi = int(black) + smooth, max(p95, int(black) + 3 * smooth)
    window = counts[lo:hi]
    local = np.convolve(window, np.ones(smooth) / smooth, mode="same")
    sampled = local >= 100
    if sampled.sum() < 3 * smooth:
        return {"scale": float("nan"), "populated_fraction": float("nan"), "codes": int(sampled.sum())}
    populated = float((window[sampled] > 0.3 * local[sampled]).mean())
    return {"scale": 1 / populated, "populated_fraction": populated, "codes": int(sampled.sum())}


def plot(
    table: pd.DataFrame, fits: dict, pairs: dict, clip: dict, histogram: np.ndarray, title: str, path: Path
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"G2 (0,0)": "#008300", "B (0,1)": "#2a78d6", "R (1,0)": "#e34948", "G1 (1,1)": "#1baf7a"}
    markers = "osD^v<>ph"
    sequences = list(pd.unique(table.sequence))
    fig = plt.figure(figsize=(12, 11))
    outer = fig.add_gridspec(2, 1, height_ratios=[4, 1.6], hspace=0.22)
    upper = outer[0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    lower = outer[1].subgridspec(1, 2, wspace=0.25)
    ax = fig.add_subplot(upper[0])
    ax_res = fig.add_subplot(upper[1], sharex=ax)
    ax_pairs = fig.add_subplot(lower[0])
    ax_hist = fig.add_subplot(lower[1])

    lo, hi = table.level.quantile([0.001, 0.999])
    bins = np.arange(np.floor(lo / 50) * 50, hi + 50, 50)
    for name, color in colors.items():
        fit = fits[name]
        slope = 1 / fit["gain"]
        for s, sequence in enumerate(sequences):
            group = table[(table.channel == name) & (table.sequence == sequence)]
            if group.empty:
                continue
            idx = np.digitize(group.level, bins)
            centres, medians, errors = [], [], []
            for k in range(1, len(bins)):
                sel = idx == k
                if sel.sum() > 30:
                    centres.append(group.level[sel].median())
                    medians.append(group.variance[sel].median())
                    errors.append(1.25 * group.variance[sel].std() / np.sqrt(sel.sum()))
            if not centres:
                continue
            centres, medians, errors = map(np.array, (centres, medians, errors))
            intercept = fit["intercepts"][sequence]
            label = None
            if s == 0:
                label = f"{name}: g = {fit['gain']:.3f} ± {fit['gain_err']:.3f} e-/ADU"
            marker = markers[s % len(markers)]
            ax.errorbar(centres, medians, errors, fmt=marker, ms=4, color=color, label=label, alpha=0.8)
            xx = np.linspace(centres.min() - 60, centres.max() + 60, 5)
            ax.plot(xx, slope * xx + intercept, "-", color=color, lw=1)
            residual = medians - (slope * centres + intercept)
            ax_res.errorbar(centres, residual, errors, fmt=marker, ms=4, color=color, alpha=0.8)
    for s, sequence in enumerate(sequences):
        ax.plot([], [], markers[s % len(markers)], color="k", ms=4, label=sequence)
    ax.set_ylabel("temporal variance per pixel [ADU²]\nvar(B − A)/2, star-masked, 4σ-clipped blocks")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    # Scale to the fitted lines: a bin of blocks the fit rejected -- twilight,
    # a passing cloud -- can sit at fifty times the Poisson variance.
    top = max(
        (fits[name]["level_max"] / fits[name]["gain"] + max(fits[name]["intercepts"].values()))
        for name in colors
    )
    ax.set_ylim(0, 1.25 * top)
    ax_res.set_ylim(-0.08 * top, 0.08 * top)
    ax_res.axhline(0, color="k")
    ax_res.set_ylabel("residual from the\nsequence's line")
    ax_res.set_xlabel("raw level L [ADU], black included")
    ax_res.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), visible=False)

    all_pairs = pd.concat([pairs[name]["pairs"].assign(channel=name) for name in colors], ignore_index=True)
    has_drift = np.isfinite(all_pairs.drift).any()
    if len(all_pairs):
        for s, sequence in enumerate(sequences):
            for name, color in colors.items():
                group = all_pairs[(all_pairs.sequence == sequence) & (all_pairs.channel == name)]
                x = group.drift if has_drift else np.arange(len(group))
                ax_pairs.plot(x, group.gain, markers[s % len(markers)], color=color, ms=4, alpha=0.7)
        if has_drift and len(sequences) == 1:
            for name, color in colors.items():
                fit = pairs[name]
                if "gain_zero_drift" not in fit:
                    continue
                xx = np.linspace(0, all_pairs.drift.max(), 5)
                ax_pairs.plot(
                    xx, fit["gain_per_pixel_drift"] * xx + fit["gain_zero_drift"], "-", color=color, lw=1
                )
                ax_pairs.plot(
                    [], [], "-", color=color, label=f"{name}: g(drift→0) = {fit['gain_zero_drift']:.3f}"
                )
            ax_pairs.legend(fontsize=8)
        p10, p90 = all_pairs.gain.quantile([0.1, 0.9])
        ax_pairs.set_ylim(max(0, p10 - 0.4), p90 + 0.4)
    ax_pairs.set_xlabel(
        "pointing drift between the two frames [px]" if has_drift else "pair (no WCS, drift unknown)"
    )
    ax_pairs.set_ylabel("per-pair gain, per channel [e-/ADU]")
    ax_pairs.grid(alpha=0.3)

    c = clip["clip"] or ADC_LEVELS - 1
    values = np.arange(max(c - 180, clip["white_level"] - 20), min(c + 120, ADC_LEVELS))
    ax_hist.semilogy(values, np.maximum(histogram[values], 0.5), drawstyle="steps-mid", color="k")
    if clip["clip"]:
        ax_hist.axvline(c, color="r", ls=":")
        clip_label = f" clip = {c}\n ({clip['pixels_at_clip']} px)"
        if clip.get("mixed_clip_states"):
            clip_label += "\n mixed states, see clip_by_sequence"
        ax_hist.text(c, histogram[c] * 0.5, clip_label, color="r", fontsize=8, va="top")
    else:
        ax_hist.text(
            0.98,
            0.95,
            "no saturation pile",
            transform=ax_hist.transAxes,
            color="r",
            fontsize=8,
            ha="right",
            va="top",
        )
    if values[0] <= clip["white_level"]:
        ax_hist.axvline(clip["white_level"], color="gray", ls=":")
        ax_hist.text(clip["white_level"], 1, " WHTLVLN", color="gray", fontsize=8, rotation=90, va="bottom")
    ax_hist.set_xlabel("raw ADU")
    ax_hist.set_ylabel("pixel count, all frames")
    ax_hist.text(0.02, 0.95, "top of the raw histogram", transform=ax_hist.transAxes, fontsize=9, va="top")
    if np.isfinite(clip.get("scale", np.nan)):
        scale_label = (
            f"raw scaled by {clip['scale']:.2f}: {1 - clip['populated_fraction']:.0%} of codes empty"
            if clip["scale"] > 1.05
            else "no missing codes: raw unscaled"
        )
        ax_hist.text(
            0.02,
            0.85,
            scale_label,
            transform=ax_hist.transAxes,
            fontsize=8,
            va="top",
            color="r" if clip["scale"] > 1.05 else "gray",
        )
    ax_hist.grid(alpha=0.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    frames: list[FrameInfo] = []
    for directory in args.sequences:
        directory = Path(directory)
        paths = sorted(p for p in directory.iterdir() if p.name.endswith((".fits", ".fits.fz")))
        if args.max_frames:
            paths = paths[: args.max_frames]
        frames.extend(read_header(p, directory.name, n) for n, p in enumerate(paths))
    if not frames:
        print("No frames found.", file=sys.stderr)
        return 1

    cameras = {f.camera for f in frames}
    isos = {f.iso for f in frames}
    units = {f.unit for f in frames}
    if len(cameras) > 1 or len(isos) > 1:
        print(f"Refusing to mix cameras {cameras} or ISOs {isos}: one camera, one ISO", file=sys.stderr)
        return 1
    camera, iso = cameras.pop(), isos.pop()
    unit = units.pop() if len(units) == 1 else "/".join(sorted(u or "?" for u in units))
    white_level = int(frames[0].white_level or args.white_level)
    print(f"{len(frames)} frames, {len(args.sequences)} sequence(s); unit {unit}, camera {camera}, ISO {iso}")

    blocks_path, histogram_path = output / "ptc_blocks.csv.gz", output / "histogram.npy"
    reuse = args.reuse and blocks_path.exists() and histogram_path.exists()
    jobs = []
    offsets_by_sequence = {}
    for sequence in dict.fromkeys(f.sequence for f in frames):
        seq_frames = [f for f in frames if f.sequence == sequence]
        mask_path = output / f"starmask_{sequence}.npy"
        if not reuse:
            template = build_template([f.path for f in seq_frames], args.template_every, args.template_frames)
            mask = star_mask(template, args.star_threshold, args.star_dilate)
            np.save(mask_path, mask)
            print(f"  {sequence}: {len(seq_frames)} frames, star mask covers {mask.mean():.1%} of pixels")
        offsets_by_sequence[sequence] = pointing_offsets(seq_frames)
        for a, b in zip(seq_frames, seq_frames[args.pair_step :]):
            jobs.append(
                (sequence, a.index, a.path, b.path, str(mask_path), args.block, args.clip_margin, white_level)
            )

    skipped: list[str] = []
    if reuse:
        print(f"reusing {blocks_path} and {histogram_path}")
        table = pd.read_csv(blocks_path)
        histogram = np.load(histogram_path)
        table = table.rename(columns={"phase": "channel"})
    else:
        histogram = np.zeros(ADC_LEVELS, dtype=np.int64)
        rows = []
        with ProcessPoolExecutor(args.jobs) as pool:
            for job, (pair_rows, pair_hist) in zip(jobs, pool.map(pair_variance, jobs, chunksize=1)):
                if not pair_rows:
                    skipped.append(f"{job[0]}:{job[1]}")
                rows.extend(pair_rows)
                histogram += pair_hist
        if skipped:
            print(f"skipped {len(skipped)} pair(s) with an unreadable frame: {', '.join(skipped)}")
        columns = ["sequence", "pair", "channel", "block", "level", "variance", "fraction"]
        table = pd.DataFrame(rows, columns=columns)
        table["pair_id"] = table.sequence + ":" + table.pair.astype(str)
        table = table[table.fraction > args.min_fraction]
        table.to_csv(blocks_path, index=False)
        np.save(histogram_path, histogram)

    drift_by_pair = {}
    for sequence, offsets in offsets_by_sequence.items():
        step = np.hypot(*(offsets[args.pair_step :] - offsets[: -args.pair_step]).T)
        drift_by_pair.update({f"{sequence}:{i}": d for i, d in enumerate(step)})
    drift_series = pd.Series(drift_by_pair, dtype=float)

    fits = {name: fit_group(table[table.channel == name], args.black) for name in CHANNELS}
    fits["all channels"] = fit_group(table, args.black)
    pairs = {name: fit_pairs(table[table.channel == name], drift_series, args.min_range) for name in CHANNELS}
    pairs["all channels"] = fit_pairs(table, drift_series, args.min_range)
    clip = find_clip(histogram, white_level)
    clip.update(find_scale(histogram, args.black, clip["clip"] or ADC_LEVELS))
    # Clip and scale are per-frame states, and a pooled histogram of a mixed
    # set reports whichever state has the bigger pile: one mid-sequence frame
    # per sequence says which state each sequence is in. The scale is read on
    # one Bayer channel, because the empty codes of the four channels sit at
    # different positions and the comb of a whole frame is partly filled in.
    clip_by_sequence, scale_by_sequence = {}, {}
    for sequence in dict.fromkeys(f.sequence for f in frames):
        seq_frames = [f for f in frames if f.sequence == sequence]
        data = read_data(seq_frames[len(seq_frames) // 2].path)
        if data is None:
            continue
        frame_clip = find_clip(np.bincount(data.ravel(), minlength=ADC_LEVELS), white_level)["clip"]
        clip_by_sequence[sequence] = frame_clip
        channel_hist = np.bincount(data[0::2, 0::2].ravel(), minlength=ADC_LEVELS)
        scale_by_sequence[sequence] = find_scale(channel_hist, args.black, frame_clip or ADC_LEVELS)["scale"]
    clip["clip_by_sequence"] = clip_by_sequence
    clip["scale_by_sequence"] = scale_by_sequence
    clip["scale_all_channels"] = clip["scale"]
    if scale_by_sequence:
        clip["scale"] = float(np.nanmedian(list(scale_by_sequence.values())))
    states = sorted({c for c in clip_by_sequence.values() if c is not None})
    clip["mixed_clip_states"] = len(states) > 1
    if clip["mixed_clip_states"]:
        print(
            f"WARNING: the sequences clip at {states}; the pooled clip {clip['clip']} is whichever state "
            "piled higher. Read clip_by_sequence, and measure the states separately for the registry.",
            file=sys.stderr,
        )
    pair_tables = [p["pairs"].assign(channel=name) for name, p in pairs.items() if len(p["pairs"])]
    if pair_tables:
        pd.concat(pair_tables, ignore_index=True).to_csv(output / "pairs.csv", index=False)

    summary = {
        "unit": unit,
        "camera": camera,
        "iso": iso,
        "sequences": list(dict.fromkeys(f.sequence for f in frames)),
        "num_frames": len(frames),
        "exptime": sorted({f.exptime for f in frames}),
        "black_assumed": args.black,
        "block": args.block,
        "fits": fits,
        "pairs": {name: {k: v for k, v in p.items() if k != "pairs"} for name, p in pairs.items()},
        "clip": clip,
        "skipped_pairs": skipped,
    }
    (output / "gain.json").write_text(json.dumps(summary, indent=2))

    print(
        (
            f"\nclip {clip['clip']} ADU ({clip['pixels_at_clip']} px at it, "
            f"{clip['pixels_above_clip']} above)"
            if clip["clip"]
            else "\nclip: none (no frame saturated: no pile above WHTLVLN)"
        )
        + f"; WHTLVLN {white_level}; raw scale {clip['scale']:.2f} (median over sequences; "
        f"{1 - clip['populated_fraction']:.0%} of sky codes empty over {clip['codes']} codes, all channels)"
    )
    print(
        "per sequence: "
        + ", ".join(
            f"{k} clip {clip_by_sequence.get(k) or 'none'} scale {v:.2f}"
            for k, v in scale_by_sequence.items()
        )
    )
    print(
        f"{'set':<12}{'blocks':>8}{'level':>14}{'gain':>8}{'±':>7}{'1-line':>8}{'black(int)':>11}{'RN e-':>7}"
        f"{'pairs':>7}{'median':>8}{'p10-p90':>14}{'g(drift→0)':>12}{'r':>6}"
    )
    for name, fit in fits.items():
        p = pairs[name]
        black_int = np.median(list(fit["black_intercept"].values()))
        read_noise = np.median(list(fit["read_noise_e"].values()))
        at_zero = f"{p['gain_zero_drift']:>12.3f}{p['correlation']:>6.2f}" if "gain_zero_drift" in p else ""
        spread = f"{p['gain_p10']:.3f}-{p['gain_p90']:.3f}" if p["num_pairs"] else ""
        print(
            f"{name:<12}{fit['num_blocks']:>8}{fit['level_min']:>7.0f}-{fit['level_max']:<6.0f}"
            f"{fit['gain']:>8.3f}{fit['gain_err']:>7.3f}{fit['single_line']['gain']:>8.3f}{black_int:>11.0f}"
            f"{read_noise:>7.1f}{p['num_pairs']:>7}{p.get('gain_median', float('nan')):>8.3f}"
            f"{spread:>14}{at_zero}"
        )
    print("\nper sequence, all channels (per-pair gains; a wide p10-p90 is cloud or haze, not the camera):")
    print(
        f"{'sequence':<18}{'pairs':>6}{'median':>8}{'p10-p90':>14}{'drift px':>10}"
        f"{'black(int)':>11}{'RN e-':>7}"
    )
    pooled = fits["all channels"]
    for name, p in pairs["all channels"].get("per_sequence", {}).items():
        print(
            f"{name:<18}{p['num_pairs']:>6}{p['gain_median']:>8.3f}{p['gain_p10']:>7.3f}-{p['gain_p90']:<6.3f}"
            f"{p['median_drift']:>10.2f}{pooled['black_intercept'][name]:>11.0f}{pooled['read_noise_e'][name]:>7.1f}"
        )
    print("gain: shared slope with one intercept per sequence; 1-line: a single intercept across sequences;")
    print(
        f"black(int): the intercept read as black with RN = 0; RN e-: with black fixed at {args.black:.0f}."
    )
    if not args.no_plot:
        title = f"Photon transfer, {unit} / {camera}, ISO {iso}: {', '.join(summary['sequences'])}"
        plot(table, fits, pairs, clip, histogram, title, output / "gain.png")
    print(f"\nwrote {output}/gain.json, ptc_blocks.csv.gz, pairs.csv, gain.png")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sequences", nargs="+", help="Raw sequence directories, one camera and ISO")
    parser.add_argument("--output", required=True, help="Directory for the tables, gain.json and gain.png")
    parser.add_argument("--block", type=int, default=64, help="Block side in pixels of one Bayer channel")
    parser.add_argument("--pair-step", type=int, default=1, help="Frames between the two of a pair")
    parser.add_argument("--black", type=float, default=2048.0, help="Black level assumed for the read noise")
    parser.add_argument("--white-level", type=int, default=11765, help="Fallback when WHTLVLN is absent")
    parser.add_argument("--clip-margin", type=int, default=1000, help="Drop pixels within this of WHTLVLN")
    parser.add_argument("--min-fraction", type=float, default=0.5, help="Drop blocks with less usable")
    parser.add_argument(
        "--min-range", type=float, default=150.0, help="Sky range a pair needs for its own fit"
    )
    parser.add_argument("--star-threshold", type=float, default=4.0, help="Star mask threshold in sky sigma")
    parser.add_argument("--star-dilate", type=int, default=3, help="Grow the star mask by this many pixels")
    parser.add_argument("--template-every", type=int, default=4, help="Use every Nth frame for the template")
    parser.add_argument("--template-frames", type=int, default=32, help="At most this many template frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Use only the first N frames per sequence")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="Refit and replot from a previous run's tables")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
