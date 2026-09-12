"""Transit injection and recovery.

The only honest way to show an algorithm change helped is to inject a known
signal and measure how much of it comes back. Self-subtraction is a real risk
here: the comparison star is assembled from stars chosen for their similarity
to the target over the *whole* sequence, in-transit frames included, so a deep
enough transit can partially reproduce itself in its own comparison.

Flux marginalisation (core.normalize_psc) protects against most of this by
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

    Injection happens at the pixel level, *before* normalisation and reference
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
        safe (improvement plan 3.6).
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

    return RecoveryResult(
        injected_depth=injected_depth,
        recovered_depth=float(out_level - in_level),
        out_of_transit_rms=float(np.nanstd(flux[out_transit])) if out_transit.any() else np.nan,
        num_in_transit=int(in_transit.sum()),
    )
