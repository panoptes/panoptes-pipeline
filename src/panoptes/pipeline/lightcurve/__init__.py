"""Differential photometry for Bayer CFA postage stamp cubes.

A pure-array implementation of the algorithm published in Gee et al., *On-sky
Demonstration of Precision Photometry with Bayer Color Filter Arrays*, with no
I/O, no cloud dependencies and no notebook required.

Typical use::

    from panoptes.pipeline.lightcurve import make_lightcurve, metrics

    result = make_lightcurve(target_psc, reference_pool, num_refs=100)
    scorecard = metrics.report(times, result.flux)
"""

from panoptes.pipeline.lightcurve import injection, masks, metrics
from panoptes.pipeline.lightcurve.core import (
    LightcurveResult,
    build_comparison,
    central_sky_mask,
    differential_lightcurve,
    make_lightcurve,
    normalize_psc,
    select_references,
    similarity_scores,
    solve_coefficients,
    subtract_stamp_sky,
)

__all__ = [
    "LightcurveResult",
    "build_comparison",
    "central_sky_mask",
    "differential_lightcurve",
    "injection",
    "make_lightcurve",
    "masks",
    "metrics",
    "normalize_psc",
    "select_references",
    "similarity_scores",
    "solve_coefficients",
    "subtract_stamp_sky",
]
