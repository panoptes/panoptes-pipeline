"""Transit detection statistics -- the objective the pipeline is actually for.

Scatter is a proxy. The figure of merit for a transit survey is how reliably a
transit of a given depth and duration is recovered at a fixed false-alarm rate,
and a change that lowers RMS while lowering completeness is a regression. See
algorithm design 4.

Nothing here depends on how the lightcurve was produced, so a new architecture
can be scored against the old one on identical terms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from panoptes.pipeline.lightcurve.metrics import to_seconds


def transit_weights(
    times: np.ndarray,
    mid_transit: float,
    duration_hours: float,
    ingress_fraction: float = 0.15,
) -> np.ndarray:
    """Trapezoidal in-transit weight per point, 1.0 at full depth."""
    seconds = to_seconds(times)
    half = duration_hours * 3600.0 / 2.0
    ramp = max(half * 2.0 * ingress_fraction, 1e-9)
    offset = np.abs(seconds - mid_transit)
    return np.clip(np.minimum(1.0, (half - offset) / ramp), 0.0, 1.0)


def fit_depth(
    flux: np.ndarray,
    weights: np.ndarray,
    sigma: np.ndarray | float | None = None,
) -> tuple[float, float]:
    """Weighted least-squares depth and its uncertainty, with a free baseline.

    Models ``flux = baseline - depth * weights``. Fitting the baseline rather
    than assuming 1.0 matters: a lightcurve normalised by its median is not
    exactly at unity once a transit is present in the data being normalised.

    Returns:
        ``(depth, depth_error)``. Depth is positive for a dip.
    """
    flux = np.asarray(flux, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(flux) & np.isfinite(weights)

    if sigma is None:
        out = good & (weights <= 0)
        scatter = _robust_sigma(flux[out]) if out.sum() > 3 else _robust_sigma(flux[good])
        sigma = scatter if scatter > 0 else 1.0

    inv_var = np.zeros_like(flux)
    inv_var[good] = 1.0 / np.asarray(sigma, dtype=float) ** 2

    # Design matrix [1, -w]; solve the 2x2 normal equations.
    s_ww = float(np.sum(inv_var * weights * weights))
    s_w = float(np.sum(inv_var * weights))
    s_1 = float(np.sum(inv_var))
    s_fw = float(np.sum(inv_var * flux * weights))
    s_f = float(np.sum(inv_var * flux))

    det = s_1 * s_ww - s_w * s_w
    if det <= 0 or s_ww <= 0:
        return 0.0, float("inf")

    depth = (s_w * s_f - s_1 * s_fw) / det
    error = float(np.sqrt(s_1 / det))
    return float(depth), error


def transit_snr(
    times: np.ndarray,
    flux: np.ndarray,
    mid_transit: float,
    duration_hours: float,
    sigma: np.ndarray | float | None = None,
) -> float:
    """Matched-filter significance of a transit at a known ephemeris."""
    weights = transit_weights(times, mid_transit, duration_hours)
    if weights.max() <= 0:
        return 0.0
    depth, error = fit_depth(flux, weights, sigma=sigma)
    return 0.0 if not np.isfinite(error) or error == 0 else depth / error


@dataclass
class Detection:
    """Best candidate found by a blind scan."""

    snr: float
    depth: float
    mid_transit: float
    duration_hours: float


def scan(
    times: np.ndarray,
    flux: np.ndarray,
    durations_hours: tuple[float, ...] = (1.0, 2.0, 4.0),
    num_phases: int = 60,
) -> Detection:
    """Blind search over mid-transit time and duration; return the best candidate.

    Scoring at the best grid point rather than the injected ephemeris is what a
    real search does, so completeness measured this way includes the penalty for
    not knowing where to look.
    """
    seconds = to_seconds(times)
    span = float(seconds[-1] - seconds[0])
    best = Detection(snr=0.0, depth=0.0, mid_transit=float(seconds[0]), duration_hours=0.0)

    for duration in durations_hours:
        half = duration * 3600.0 / 2.0
        if 2 * half > span:
            continue
        for centre in np.linspace(seconds[0] + half, seconds[-1] - half, num_phases):
            weights = transit_weights(seconds, centre, duration)
            if weights.max() <= 0:
                continue
            depth, error = fit_depth(flux, weights)
            if not np.isfinite(error) or error == 0:
                continue
            snr = depth / error
            if snr > best.snr:
                best = Detection(snr, depth, float(centre), duration)
    return best


def false_alarm_threshold(
    times: np.ndarray,
    flux: np.ndarray,
    durations_hours: tuple[float, ...] = (1.0, 2.0, 4.0),
    num_phases: int = 60,
    false_alarm_rate: float = 0.01,
    num_trials: int = 200,
    seed: int = 0,
) -> float:
    """SNR threshold giving the requested false-alarm rate on signal-free data.

    The null is built by **circular time shifts**, not by shuffling. Shuffling
    destroys the autocorrelation, which would set the threshold as if the noise
    were white and make every red-noise lightcurve look more significant than it
    is. Circular shifts preserve the correlation structure and break only the
    phase, which is the property being tested.
    """
    flux = np.asarray(flux, dtype=float)
    rng = np.random.default_rng(seed)
    count = len(flux)
    if count < 8:
        return float("inf")

    peaks = []
    for shift in rng.integers(1, count, size=num_trials):
        peaks.append(
            scan(times, np.roll(flux, int(shift)), durations_hours, num_phases).snr
        )
    return float(np.quantile(peaks, 1.0 - false_alarm_rate))


def completeness(
    recover: Callable[[float, float], np.ndarray],
    times: np.ndarray,
    depths: tuple[float, ...],
    durations_hours: tuple[float, ...],
    threshold: float,
    num_phases: int = 8,
    scan_durations: tuple[float, ...] | None = None,
) -> dict[tuple[float, float], float]:
    """Fraction of injected transits recovered above ``threshold``.

    Args:
        recover: ``recover(depth, duration_hours) -> flux``. The pipeline under
            test injects at the pixel level and returns the lightcurve it
            produces. Injecting into a finished lightcurve instead measures
            nothing about the pipeline.
        times: Observation times, matching what ``recover`` returns.
        depths: Fractional depths to inject.
        durations_hours: Durations to inject.
        threshold: From :func:`false_alarm_threshold`.
        num_phases: Injections per (depth, duration) cell.
        scan_durations: Durations the blind scan searches. Defaults to the
            injected set; pass a different grid to include the penalty for a
            search that does not know the true duration.

    Returns:
        ``{(depth, duration): recovered fraction}``.
    """
    searched = scan_durations or durations_hours
    grid = {}
    for depth in depths:
        for duration in durations_hours:
            hits = 0
            for _ in range(num_phases):
                flux = recover(depth, duration)
                hits += scan(times, flux, searched).snr >= threshold
            grid[(depth, duration)] = hits / num_phases
    return grid


def _robust_sigma(values: np.ndarray) -> float:
    """MAD-based scatter, so a transit in the sample does not inflate it."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))
