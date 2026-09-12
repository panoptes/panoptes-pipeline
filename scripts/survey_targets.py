# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
#   "scikit-learn",
#   "tables",
# ]
# ///
"""Score algorithm variants across many targets, not one.

`benchmark_lightcurve.py` answers "what happened to this star". This answers
"what happens in general", which is the only question that justifies a change.
It reports the distribution of scatter across a sample of targets and the
fraction of them each variant actually wins on -- a median improvement that
only holds for half the sample is not an improvement.

Scoring is on held-out frames by default: references are selected and
coefficients fitted using alternate frames only, then scatter is measured on
the frames the fit never saw. With 100 free coefficients per target that check
is not optional.

Usage::

    uv run scripts/survey_targets.py notebooks/PAN007_.../observation.h5
    uv run scripts/survey_targets.py OBS.h5 --channel all --num-targets 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_lightcurve import load_observation  # noqa: E402

from panoptes.pipeline.lightcurve import core, masks, metrics  # noqa: E402

VARIANT_NAMES = ("raw", "ensemble_scaled", "ols", "ridge")


def run_target(target, pool, pixel_mask, num_refs, frame_weights):
    """Return ``({variant: flux}, had_identical_twin)`` for one target."""
    target_norm = core.normalize_psc(target)
    pool_norm = core.normalize_psc(pool)
    scores = core.similarity_scores(target_norm, pool_norm, frame_weights=frame_weights)

    # A score of exactly zero means a duplicate source made it into the pool --
    # see conformance audit 5.10. Demote rather than abort, and count it.
    twin = bool(np.any(scores == 0))
    chosen = core.select_references(np.where(scores == 0, np.inf, scores), num_refs)
    refs_raw, refs_norm = pool[chosen], pool_norm[chosen]

    raw = np.where(pixel_mask, target, 0.0).sum(axis=1)
    curves = {"raw": raw / np.median(raw)}

    target_median = np.median(target.sum(axis=1))
    ref_medians = np.median(refs_raw.sum(axis=2), axis=1)
    ensemble = (refs_raw * (target_median / ref_medians)[:, None, None]).mean(axis=0)
    curves["ensemble_scaled"] = core.differential_lightcurve(
        target, ensemble, pixel_mask=pixel_mask
    )

    for name, kwargs in (("ols", dict(method="ols")), ("ridge", dict(method="ridge", alpha=1e-4))):
        coefficients = core.solve_coefficients(
            target_norm, refs_norm, frame_weights=frame_weights, **kwargs
        )
        curves[name] = core.differential_lightcurve(
            target, core.build_comparison(refs_raw, coefficients), pixel_mask=pixel_mask
        )

    return curves, twin


def run(args: argparse.Namespace) -> int:
    cube, picids, frames, stamp_shape = load_observation(args.observation)
    cube = cube.astype(np.float32)
    print(f"  {cube.shape[0]} sources x {cube.shape[1]} frames, stamp_shape={stamp_shape}")

    rgb_flat = {name: mask.ravel() for name, mask in masks.rgb_masks(stamp_shape).items()}
    if args.sky_subtract:
        cube = core.subtract_stamp_sky(cube, core.central_sky_mask(stamp_shape), rgb_flat)
        print("  sky pedestal removed (stopgap, see conformance audit 5.0)")
    else:
        print("  WARNING: stamps carry an un-subtracted sky pedestal; results are diluted")

    if args.channel == "all":
        pixel_mask = np.ones(cube.shape[2], dtype=bool)
    else:
        pixel_mask = rgb_flat[args.channel]

    weights, keep = None, np.ones(len(frames), dtype=bool)
    if not args.in_sample:
        weights = np.zeros(len(frames))
        weights[::2] = 1.0
        keep = weights == 0

    # A contiguous band at the bright end. Spreading the sample across all
    # sources instead pulls in stars with no measurable flux once the sky is
    # removed, and fractional RMS on near-zero flux is not a precision metric.
    ranked = np.argsort(np.median(np.where(pixel_mask, cube, 0.0).sum(axis=2), axis=1))[::-1]
    sample = ranked[args.skip_brightest : args.skip_brightest + args.num_targets]
    sample_flux = np.median(np.where(pixel_mask, cube[sample], 0.0).sum(axis=2))
    print(f"  sample median flux in band: {sample_flux:.3e} ADU")

    results = {name: [] for name in VARIANT_NAMES}
    twins = 0
    for index in sample:
        try:
            curves, twin = run_target(
                cube[index],
                np.delete(cube, index, axis=0),
                pixel_mask,
                args.num_refs,
                weights,
            )
        except Exception as error:  # noqa: BLE001 - one bad target must not stop the survey
            print(f"  skipped PICID {picids[index]}: {error!r}"[:100])
            continue
        twins += twin
        for name in VARIANT_NAMES:
            results[name].append(metrics.rms(curves[name][keep]))

    _print_summary(results, twins, args, int(keep.sum()), len(frames))
    return 0


def _print_summary(results, twins, args, num_scored, num_frames) -> None:
    arrays = {name: np.array(values) for name, values in results.items()}
    finite = np.ones(len(arrays["ols"]), dtype=bool)
    for values in arrays.values():
        finite &= np.isfinite(values)

    mode = "in-sample" if args.in_sample else f"held-out ({num_scored} of {num_frames} frames)"
    print(f"\n=== channel={args.channel} | {finite.sum()} targets | {mode} ===")
    print(f"{'variant':<18}{'median':>9}{'p25':>9}{'p75':>9}")
    for name in VARIANT_NAMES:
        values = arrays[name][finite]
        print(
            f"{name:<18}{np.median(values):>8.2%} "
            f"{np.percentile(values, 25):>8.2%} {np.percentile(values, 75):>8.2%}"
        )

    print()
    for baseline in ("raw", "ensemble_scaled"):
        for name in ("ols", "ridge"):
            wins = np.mean(arrays[name][finite] < arrays[baseline][finite])
            gain = np.median(arrays[baseline][finite] / arrays[name][finite])
            print(f"{name} beats {baseline:<16}: {wins:>4.0%} of targets, median gain {gain:.2f}x")

    if twins:
        print(
            f"\n{twins} target(s) had an identical-stamp twin in the reference pool "
            "(duplicate catalog match -- conformance audit 5.10)"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("observation", type=Path, help="Path to an observation.h5")
    parser.add_argument("--channel", choices=["all", "r", "g", "b"], default="r")
    parser.add_argument("--num-targets", type=int, default=80)
    parser.add_argument("--num-refs", type=int, default=100)
    parser.add_argument(
        "--skip-brightest", type=int, default=20, help="Skip this many likely-saturated stars"
    )
    parser.add_argument(
        "--sky-subtract",
        action="store_true",
        help="Remove the per-frame, per-colour sky pedestal before running",
    )
    parser.add_argument(
        "--in-sample",
        action="store_true",
        help="Score on all frames instead of held-out frames (for measuring overfitting)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
