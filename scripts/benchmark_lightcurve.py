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
"""Benchmark differential photometry variants against a stored observation.

This is the harness improvement plan 1 is built on: it runs several algorithm
variants over the same target and prints one table of precision metrics, so a
proposed change either moves the numbers or it does not.

It reads the ``observation.h5`` produced by ``ProcessObservation.ipynb`` and
needs no network, no Firestore and no BigQuery.

Usage::

    uv run scripts/benchmark_lightcurve.py notebooks/PAN007_.../observation.h5
    uv run scripts/benchmark_lightcurve.py OBS.h5 --picid 12632720
    uv run scripts/benchmark_lightcurve.py OBS.h5 --inject-depth 0.01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from panoptes.pipeline.lightcurve import core, injection, masks, metrics  # noqa: E402

#: Variants compared on every run.
#:
#: `legacy_mean` is what `notebooks/working/MakeLightcurves.ipynb` does today:
#: an unweighted mean of the top references' *raw* stamps, with no coefficient
#: fit at all. Because the reference pool spans mV 6-13, that mean is dominated
#: by whichever reference is brightest.
#:
#: `ensemble_scaled` is the fair comparison: each reference is scaled to the
#: target's median flux before averaging, which is ordinary ensemble
#: differential photometry. Any credit claimed for the coefficient fit must be
#: measured against this, not against `legacy_mean`.
VARIANTS = {
    "legacy_mean": dict(builder="legacy_mean"),
    "ensemble_scaled": dict(builder="ensemble_scaled"),
    "paper_ols": dict(method="ols"),
    "ridge": dict(method="ridge", alpha=1e-4),
    "lasso": dict(method="lasso", alpha=1e-7),
    "nnls": dict(method="nnls"),
}


def load_observation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Read stamps into a dense ``(num_sources, num_frames, num_pixels)`` cube.

    Sources that are missing from any frame are dropped, so every PSC shares a
    single common time axis.
    """
    with pd.HDFStore(path, "r") as store:
        stamps = store["stamps"]
        try:
            attrs = store.get_storer("positions").attrs.metadata
        except (AttributeError, KeyError):
            attrs = {}
        positions = store["positions"]

    pixel_cols = [c for c in stamps.columns if c.startswith("pixel_")]
    num_pixels = len(pixel_cols)

    width = int(attrs.get("stamp_width", int(np.sqrt(num_pixels))))
    height = int(attrs.get("stamp_height", num_pixels // max(width, 1)))

    # make_stamps() slices data[y_min:y_max, x_min:x_max] and ravels row-major,
    # so the flat pixel axis unpacks as (height, width) -- NOT (width, height).
    stamp_shape = (height, width)
    if height * width != num_pixels:
        raise ValueError(f"{stamp_shape=} does not account for {num_pixels} pixels")

    frames = np.sort(stamps.index.get_level_values("time").unique())
    counts = stamps.groupby(level="picid").size()
    complete = counts[counts == len(frames)].index

    ordered = stamps.loc[stamps.index.get_level_values("picid").isin(complete)]
    ordered = ordered.sort_index(level=["picid", "time"])

    picids = np.asarray(sorted(complete))
    cube = ordered[pixel_cols].to_numpy(dtype=float)
    cube = cube.reshape(len(picids), len(frames), num_pixels)

    _check_superpixel_alignment(positions.loc[positions.index.isin(picids)], stamp_shape)

    return cube, picids, frames, stamp_shape


def _check_superpixel_alignment(positions: pd.DataFrame, stamp_shape: tuple[int, int]) -> None:
    """Warn if any stamp carries a different Bayer phase from the others."""
    if not {"stamp_x_min", "stamp_y_min"}.issubset(positions.columns):
        return
    misaligned = (positions.stamp_x_min % 2 != 0) | (positions.stamp_y_min % 2 != 0)
    height, width = stamp_shape
    if height % 2 or width % 2:
        print(f"  WARNING: {stamp_shape=} is not an integer number of superpixels")
    if misaligned.any():
        print(
            f"  WARNING: {int(misaligned.sum())} of {len(positions)} stamps have an odd origin, "
            "so their Bayer phase differs from the rest. Pixel-level comparison is invalid "
            "for these -- see conformance audit 4.3."
        )


def pick_target(cube: np.ndarray, picids: np.ndarray, requested: int | None) -> int:
    """Return the index of the requested PICID, or of a bright, unsaturated star."""
    if requested is not None:
        matches = np.flatnonzero(picids == requested)
        if not len(matches):
            raise SystemExit(f"PICID {requested} not present in this observation")
        return int(matches[0])

    total = cube.sum(axis=(1, 2))
    ranked = np.argsort(total)[::-1]
    return int(ranked[len(ranked) // 20])  # ~95th percentile: bright but not the brightest


def unfitted_lightcurve(target, pool, num_refs, builder, channel_mask=None, frame_weights=None):
    """Comparison stars built without a coefficient fit.

    ``legacy_mean`` averages the raw stamps (today's notebook behavior);
    ``ensemble_scaled`` first scales each reference to the target's median flux,
    which is conventional ensemble differential photometry.
    """
    target_norm = core.normalize_psc(target)
    pool_norm = core.normalize_psc(pool)
    scores = core.similarity_scores(target_norm, pool_norm, frame_weights=frame_weights)
    chosen = core.select_references(scores, num_refs=num_refs)
    refs = pool[chosen]

    if builder == "ensemble_scaled":
        target_median = np.median(target.sum(axis=1))
        ref_medians = np.median(refs.sum(axis=2), axis=1)
        comparison = (refs * (target_median / ref_medians)[:, None, None]).mean(axis=0)
    else:
        comparison = refs.mean(axis=0)

    flux = core.differential_lightcurve(target, comparison, pixel_mask=channel_mask)
    return core.LightcurveResult(
        flux=flux,
        comparison=comparison,
        coefficients=np.full(len(chosen), 1.0 / len(chosen)),
        reference_indices=chosen,
        scores=scores[chosen],
        meta=dict(method=builder, num_refs=num_refs),
    )


def run(args: argparse.Namespace) -> int:
    print(f"Loading {args.observation}")
    cube, picids, frames, stamp_shape = load_observation(args.observation)
    print(f"  {cube.shape[0]} complete sources x {cube.shape[1]} frames x {cube.shape[2]} pixels")
    print(f"  stamp_shape={stamp_shape}")

    rgb_flat = {name: mask.ravel() for name, mask in masks.rgb_masks(stamp_shape).items()}
    if args.sky_subtract:
        sky = core.central_sky_mask(stamp_shape)
        cube = core.subtract_stamp_sky(cube, sky, rgb_flat)
        print("  sky pedestal removed per frame and color (stopgap, see conformance audit 5.0)")
    else:
        print(
            "  WARNING: stamps used as stored. ProcessFITS.ipynb writes RAW data to\n"
            "  reduced_filename, so these carry bias + sky (~780 ADU/pixel). Every number\n"
            "  below is diluted by that pedestal. Re-run with --sky-subtract."
        )

    index = pick_target(cube, picids, args.picid)
    target = cube[index]
    pool = np.delete(cube, index, axis=0)
    print(f"  target PICID {picids[index]} ({cube.shape[0] - 1} candidate references)")

    model = None
    if args.inject_depth:
        midpoint = metrics.to_seconds(frames)[len(frames) // 2]
        model = injection.trapezoid_transit(
            frames,
            mid_transit=midpoint,
            duration_hours=args.inject_duration,
            depth=args.inject_depth,
        )
        target = injection.inject(target, model)
        print(f"  injected {args.inject_depth:.1%} transit over {args.inject_duration} h")

    channel_mask = None
    if args.channel != "all":
        channel_mask = rgb_flat[args.channel]
        print(f"  photometry restricted to the {args.channel} channel")

    # Raw aperture photometry, on the same pixels the variants will use.
    raw_pixels = target if channel_mask is None else np.where(channel_mask, target, 0.0)
    raw = raw_pixels.sum(axis=1)

    # Held-out mode: select references and fit coefficients on alternate frames
    # only, then score on the frames the fit never saw. With 100 free
    # coefficients this is the check that the gain is not overfitting.
    weights, keep = None, np.ones(len(frames), dtype=bool)
    if args.held_out:
        weights = np.zeros(len(frames))
        weights[::2] = 1.0
        keep = weights == 0
        print(f"  held-out scoring on {int(keep.sum())} of {len(frames)} frames")

    def score(flux, active=None):
        return metrics.report(frames[keep], flux[keep], num_active_references=active)

    rows = [("raw_aperture", score(raw / np.median(raw)), None)]

    for name, settings in VARIANTS.items():
        if "builder" in settings:
            result = unfitted_lightcurve(
                target,
                pool,
                args.num_refs,
                settings["builder"],
                channel_mask=channel_mask,
                frame_weights=weights,
            )
        else:
            result = core.make_lightcurve(
                target,
                pool,
                num_refs=args.num_refs,
                method=settings["method"],
                alpha=settings.get("alpha", 1e-5),
                channel_mask=channel_mask,
                channel=args.channel,
                frame_weights=weights,
            )
        card = score(result.flux, result.num_active_references)
        recovery = injection.measure_depth(result.flux, model) if model is not None else None
        rows.append((name, card, recovery))

    _print_table(rows, model is not None)
    return 0


def _print_table(rows, injected: bool) -> None:
    header = f"\n{'variant':<14}{'rms':>9}{'30min':>9}{'beta':>7}{'refs':>6}"
    if injected:
        header += f"{'depth':>9}{'suppr':>8}"
    print(header)
    print("-" * len(header.strip()))

    for name, card, recovery in rows:
        active = card.num_active_references
        line = (
            f"{name:<14}"
            f"{card.unbinned_rms:>8.2%} "
            f"{card.binned_rms.get(30.0, float('nan')):>8.2%}"
            f"{card.beta:>7.2f}"
            f"{(active if active is not None else '-'):>6}"
        )
        if injected:
            if recovery is None:
                line += f"{'-':>9}{'-':>8}"
            else:
                line += f"{recovery.recovered_depth:>8.2%} {recovery.suppression:>7.1%}"
        print(line)

    print(
        "\nrms/30min: fractional scatter unbinned and in 30 min bins (paper reports "
        "2-4% and ~1%).\nbeta: red-noise factor, 1.0 means noise averages down as "
        "white noise.\nrefs: coefficients surviving regularization (paper Fig. 7: 46 of 100)."
    )
    if injected:
        print("depth/suppr: recovered transit depth and the fraction of it lost.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("observation", type=Path, help="Path to an observation.h5")
    parser.add_argument(
        "--picid", type=int, default=None, help="Target PICID (default: a bright star)"
    )
    parser.add_argument("--num-refs", type=int, default=100, help="References carried into the fit")
    parser.add_argument("--channel", choices=["all", "r", "g", "b"], default="all")
    parser.add_argument(
        "--sky-subtract",
        action="store_true",
        help="Remove the per-frame, per-color sky pedestal the pipeline failed to subtract",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Fit on alternate frames and score only on the frames the fit never saw",
    )
    parser.add_argument(
        "--inject-depth", type=float, default=0.0, help="Inject a transit of this depth"
    )
    parser.add_argument(
        "--inject-duration", type=float, default=1.0, help="Injected transit duration [hours]"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
