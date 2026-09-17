"""Transit injection and recovery.

The only honest way to show an algorithm change helped is to inject a known
signal and measure how much of it comes back. Self-subtraction is a real risk
here: the comparison star is assembled from stars chosen for their similarity
to the target over the *whole* sequence, in-transit frames included, so a deep
enough transit can partially reproduce itself in its own comparison.

Flux marginalization (core.normalize_psc) protects against most of this by
construction, but "mostly" is not a number. These helpers produce the number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panoptes.pipeline.lightcurve.metrics import to_seconds as _to_seconds


def box_transit(
    times: np.ndarray,
    mid_transit: float,
    duration_hours: float,
    depth: float,
) -> np.ndarray:
    """Sharp-edged transit model. ``1.0`` out of transit, ``1 - depth`` inside."""
    seconds = _to_seconds(times)
    half = duration_hours * 3600.0 / 2.0
    in_transit = np.abs(seconds - mid_transit) <= half
    return np.where(in_transit, 1.0 - depth, 1.0)


def trapezoid_transit(
    times: np.ndarray,
    mid_transit: float,
    duration_hours: float,
    depth: float,
    ingress_fraction: float = 0.15,
) -> np.ndarray:
    """Trapezoidal transit with linear ingress and egress ramps.

    Closer to a real limb-darkened transit than a box while staying analytic.
    ``ingress_fraction`` is the ramp length as a fraction of total duration.
    """
    seconds = _to_seconds(times)
    half = duration_hours * 3600.0 / 2.0
    ramp = max(half * 2.0 * ingress_fraction, 1e-9)

    offset = np.abs(seconds - mid_transit)
    flat = half - ramp
    ramp_value = np.clip((half - offset) / ramp, 0.0, 1.0)
    shape = np.where(offset <= flat, 1.0, ramp_value)
    return 1.0 - depth * np.clip(shape, 0.0, 1.0)


def inject(psc: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Scale every pixel of each frame by the transit model.

    Injection happens at the pixel level, *before* normalization and reference
    selection, so the full algorithm sees the signal exactly as it would see a
    real one. Injecting into the finished lightcurve instead would measure
    nothing.

    Args:
        psc: ``(m, n)`` raw target PSC.
        model: ``(m,)`` relative flux model, 1.0 out of transit.

    Returns:
        ``(m, n)`` PSC with the transit imprinted.
    """
    psc = np.asarray(psc, dtype=float)
    model = np.asarray(model, dtype=float)
    if model.shape[0] != psc.shape[0]:
        raise ValueError(f"Model has {model.shape[0]} frames, PSC has {psc.shape[0]}")
    return psc * model[:, None]


@dataclass
class RecoveryResult:
    """Outcome of one injection/recovery trial."""

    injected_depth: float
    recovered_depth: float
    out_of_transit_rms: float
    num_in_transit: int

    @property
    def suppression(self) -> float:
        """Fraction of the injected depth lost to the algorithm. 0.0 is perfect.

        Positive values mean the comparison star partly absorbed the signal.
        This is the number that decides whether a reference-selection change is
        safe: references are chosen by similarity across every frame, in-transit
        frames included, so nothing but flux marginalization stops the
        comparison from learning the transit.
        """
        if self.injected_depth == 0:
            return float("nan")
        return 1.0 - (self.recovered_depth / self.injected_depth)

    @property
    def significance(self) -> float:
        """Recovered depth in units of the out-of-transit scatter per point."""
        if not self.out_of_transit_rms:
            return float("nan")
        return self.recovered_depth / self.out_of_transit_rms


def measure_depth(flux: np.ndarray, model: np.ndarray, threshold: float = 0.5) -> RecoveryResult:
    """Compare in-transit and out-of-transit levels of a recovered lightcurve.

    Args:
        flux: ``(m,)`` recovered relative flux.
        model: ``(m,)`` the injected model, used to label frames.
        threshold: Frames where the model dips below this fraction of full
            depth count as in-transit. Frames on the ramps are excluded from
            both samples so partial coverage does not dilute the measurement.
    """
    flux = np.asarray(flux, dtype=float)
    model = np.asarray(model, dtype=float)

    injected_depth = float(1.0 - model.min())
    if injected_depth <= 0:
        raise ValueError("Model has no transit to recover")

    dip = (1.0 - model) / injected_depth
    in_transit = dip >= (1.0 - threshold)
    out_transit = dip <= 1e-12

    in_level = float(np.nanmedian(flux[in_transit])) if in_transit.any() else np.nan
    out_level = float(np.nanmedian(flux[out_transit])) if out_transit.any() else np.nan

    # Fractional, not absolute: lightcurves are no longer normalized to a unit
    # baseline (algorithm design 6), so the difference must be divided by the
    # out-of-transit level to be a depth at all.
    scale = out_level if np.isfinite(out_level) and out_level != 0 else np.nan

    return RecoveryResult(
        injected_depth=injected_depth,
        recovered_depth=float((out_level - in_level) / scale),
        out_of_transit_rms=(float(np.nanstd(flux[out_transit]) / scale) if out_transit.any() else np.nan),
        num_in_transit=int(in_transit.sum()),
    )


# --------------------------------------------------------------------------
# Signal fidelity -- the algorithm's own objective, independent of transits
# --------------------------------------------------------------------------


def sinusoid(
    times: np.ndarray,
    period_hours: float,
    amplitude: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Relative flux model for a sinusoidal variation of known amplitude.

    Sinusoids rather than transits, deliberately. Measuring fidelity with a
    transit shape only tells you about transit-shaped signals; a sweep over
    period measures what the algorithm does to *any* variation at that
    timescale, which is what the algorithm is actually responsible for.
    """
    seconds = _to_seconds(times)
    return 1.0 + amplitude * np.sin(2.0 * np.pi * seconds / (period_hours * 3600.0) + phase)


def measure_amplitude(times: np.ndarray, flux: np.ndarray, period_hours: float) -> float:
    """Least-squares amplitude of a known-period sinusoid in a lightcurve.

    Fits ``a + b sin(wt) + c cos(wt)`` and returns ``sqrt(b^2 + c^2)``, so the
    result is independent of the injected phase.
    """
    seconds = _to_seconds(times)
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(flux)
    if good.sum() < 4:
        return float("nan")

    omega = 2.0 * np.pi / (period_hours * 3600.0)
    design = np.column_stack(
        [np.ones(good.sum()), np.sin(omega * seconds[good]), np.cos(omega * seconds[good])]
    )
    coefficients, *_ = np.linalg.lstsq(design, flux[good], rcond=None)
    return float(np.hypot(coefficients[1], coefficients[2]))


def transfer_function(
    recover,
    times: np.ndarray,
    periods_hours: tuple[float, ...],
    amplitude: float = 0.01,
    num_phases: int = 4,
) -> dict[float, float]:
    """Fraction of an injected signal that survives the pipeline, by timescale.

    This is the algorithm's objective. Its job is to return the target's true
    relative flux -- nothing suppressed, nothing invented -- and a transfer
    function of 1.0 at every timescale of interest is what that means
    quantitatively. Detection is a property of the survey built on top, not of
    the algorithm, and optimizing the algorithm for a transit shape would bias
    it toward signals matching that prior and against everything else the data
    contains.

    A long-timescale rolloff is the failure mode to watch for: a model with many
    free parameters fitted across a whole sequence will absorb slow variation,
    and a transit-shaped test at one duration can miss it entirely.

    Args:
        recover: ``recover(model) -> flux``, where ``model`` is a relative flux
            array to imprint at the pixel level before the pipeline runs.
        times: Observation times.
        periods_hours: Timescales to probe.
        amplitude: Injected amplitude, small enough to stay in the linear regime.
        num_phases: Injections per period, averaged, to remove phase sensitivity.

    Returns:
        ``{period_hours: recovered / injected}``. 1.0 is perfect fidelity.
    """
    results = {}
    for period in periods_hours:
        recovered = []
        for phase in np.linspace(0.0, 2.0 * np.pi, num_phases, endpoint=False):
            model = sinusoid(times, period, amplitude, phase=phase)
            recovered.append(measure_amplitude(times, recover(model), period))
        results[float(period)] = float(np.nanmean(recovered) / amplitude)
    return results
