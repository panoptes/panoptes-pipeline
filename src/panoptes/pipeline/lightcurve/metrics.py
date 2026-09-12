"""Photometric precision metrics.

Nothing in improvement plan 3 can be called an improvement without a number
attached, and "standard deviation of the lightcurve" is not a sufficient
number: correlated (red) noise is what actually limits transit detection, and
it does not average down. These are the metrics every change gets scored on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Bin sizes (minutes) reported by default. The 30 min point is the one the
#: paper quotes, so it is the direct comparison against published performance.
DEFAULT_BIN_MINUTES = (0, 5, 10, 15, 30, 60)


def rms(flux: np.ndarray) -> float:
    """Fractional RMS scatter about the median, ignoring NaNs."""
    flux = np.asarray(flux, dtype=float)
    median = np.nanmedian(flux)
    if not np.isfinite(median) or median == 0:
        return float("nan")
    return float(np.nanstd(flux / median))


def bin_flux(
    times: np.ndarray,
    flux: np.ndarray,
    bin_minutes: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a lightcurve into fixed-width time bins.

    Args:
        times: ``(m,)`` observation times. Accepts ``datetime64`` or float
            seconds since an arbitrary epoch.
        flux: ``(m,)`` relative flux.
        bin_minutes: Bin width. ``0`` returns the unbinned series.

    Returns:
        ``(bin_centres_seconds, binned_flux, counts_per_bin)``; empty bins are
        dropped.
    """
    seconds = to_seconds(times)
    flux = np.asarray(flux, dtype=float)

    if bin_minutes <= 0:
        return seconds, flux, np.ones_like(flux, dtype=int)

    width = bin_minutes * 60.0
    which = np.floor((seconds - seconds[0]) / width).astype(int)

    centres, means, counts = [], [], []
    for value in np.unique(which):
        selected = (which == value) & np.isfinite(flux)
        count = int(selected.sum())
        if count == 0:
            continue
        centres.append(float(seconds[selected].mean()))
        means.append(float(flux[selected].mean()))
        counts.append(count)

    return np.array(centres), np.array(means), np.array(counts, dtype=int)


def precision_curve(
    times: np.ndarray,
    flux: np.ndarray,
    bin_minutes: tuple[float, ...] = DEFAULT_BIN_MINUTES,
) -> dict[float, float]:
    """RMS as a function of bin size -- the headline diagnostic for this work.

    Pure white noise falls as ``1/sqrt(N)``. Departure from that line is the
    red noise that limits transit detection, and closing that gap is what
    improvement plan 3 is for.
    """
    return {
        float(width): rms(bin_flux(times, flux, width)[1]) for width in bin_minutes
    }


def beta_factor(
    times: np.ndarray,
    flux: np.ndarray,
    bin_minutes: float = 30.0,
) -> float:
    """Red-noise factor beta, following Pont, Zucker & Queloz (2006).

    ``beta = sigma_binned / sigma_expected``, where ``sigma_expected`` is what
    pure white noise would give after binning. ``beta == 1`` means no
    correlated noise; the paper's own result implies ``beta`` close to 1 at
    30 min, so any change that pushes it above 1 is a regression even if the
    unbinned RMS improves.
    """
    unbinned = rms(flux)
    _, binned, counts = bin_flux(times, flux, bin_minutes)

    num_bins = len(binned)
    if num_bins < 2 or not np.isfinite(unbinned) or unbinned == 0:
        return float("nan")

    mean_per_bin = float(np.mean(counts))
    if mean_per_bin <= 1:
        return float("nan")

    expected = unbinned / np.sqrt(mean_per_bin) * np.sqrt(num_bins / (num_bins - 1))
    if expected == 0:
        return float("nan")
    return float(rms(binned) / expected)


def photon_noise_floor(
    total_counts_adu: np.ndarray | float,
    gain_e_per_adu: float = 1.5,
    background_counts_adu: np.ndarray | float = 0.0,
    read_noise_e: float = 10.47,
    num_pixels: int = 1,
) -> float:
    """Fractional noise floor from source photons, sky and read noise.

    Defaults are the Canon EOS 100D values in paper Table 2. Every precision
    claim should be quoted as a multiple of this floor, not in isolation --
    the paper's stated goal is to *approach* it.
    """
    source_e = np.asarray(total_counts_adu, dtype=float) * gain_e_per_adu
    sky_e = np.asarray(background_counts_adu, dtype=float) * gain_e_per_adu
    variance = source_e + sky_e + num_pixels * read_noise_e**2
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sqrt(variance) / source_e)


@dataclass
class PrecisionReport:
    """Scorecard for one lightcurve. Compare these across algorithm variants."""

    unbinned_rms: float
    binned_rms: dict[float, float]
    beta: float
    num_frames: int
    num_active_references: int | None = None
    noise_floor: float | None = None

    @property
    def floor_ratio(self) -> float:
        """Unbinned RMS as a multiple of the photon noise floor. Lower is better."""
        if not self.noise_floor:
            return float("nan")
        return self.unbinned_rms / self.noise_floor

    def to_dict(self) -> dict:
        return dict(
            unbinned_rms=self.unbinned_rms,
            binned_rms={str(k): v for k, v in self.binned_rms.items()},
            beta=self.beta,
            num_frames=self.num_frames,
            num_active_references=self.num_active_references,
            noise_floor=self.noise_floor,
            floor_ratio=self.floor_ratio,
        )


def report(
    times: np.ndarray,
    flux: np.ndarray,
    num_active_references: int | None = None,
    noise_floor: float | None = None,
    bin_minutes: tuple[float, ...] = DEFAULT_BIN_MINUTES,
) -> PrecisionReport:
    """Build a :class:`PrecisionReport` for one lightcurve."""
    return PrecisionReport(
        unbinned_rms=rms(flux),
        binned_rms=precision_curve(times, flux, bin_minutes=bin_minutes),
        beta=beta_factor(times, flux),
        num_frames=int(np.isfinite(np.asarray(flux, dtype=float)).sum()),
        num_active_references=num_active_references,
        noise_floor=noise_floor,
    )


def to_seconds(times: np.ndarray) -> np.ndarray:
    """Coerce times to float seconds measured from the first sample.

    Accepts ``datetime64`` arrays, float arrays, and object arrays of anything
    with a ``timestamp()`` method -- which covers the timezone-aware
    ``pandas.Timestamp`` values that come out of ``observation.h5``.
    """
    times = np.asarray(times)

    if np.issubdtype(times.dtype, np.datetime64):
        return (times - times[0]) / np.timedelta64(1, "s")

    if times.dtype == object:
        first = times.flat[0]
        if hasattr(first, "timestamp"):
            epoch = np.array([t.timestamp() for t in times.ravel()], dtype=float)
            return (epoch - epoch[0]).reshape(times.shape)

    return times.astype(float)


#: Retained under the old private name so existing callers keep working.
_to_seconds = to_seconds
