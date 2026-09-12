# panoptes-pipeline

Differential photometry for PANOPTES: recovering exoplanet transits from DSLR
wide-field images with a Bayer colour filter array.

## The idea

A wide field gives any target many candidate reference stars whose light lands
on the colour filter array the same way the target's does. Those references
build an idealised star that behaves as the target would absent a transit, so
the difference is zero unless the target's flux genuinely changed.

The load-bearing insight: dividing a stamp by its own summed flux marginalises
brightness out and leaves the spatial profile. Shape and brightness separate.
Match on shape, difference the brightness.

**That idea is the only fixed point.** Gee et al., *On-sky Demonstration of
Precision Photometry with Bayer Color Filter Arrays*, is earlier research and a
building block -- not a specification. It likely contains errors. Where its
choices and the idea diverge, the idea wins, and rewriting from scratch is on
the table.

## Read these first

- `plans/algorithm-design.md` ("algorithm design") -- what the algorithm is,
  independent of implementation, and the proposed architecture. **Start here.**
  Its section 5 says which measurements are solid and which are provisional.
- `plans/conformance-audit.md` ("conformance audit") -- defects in the *existing*
  implementation. Historical context for a rebuild, not a task list.
- `plans/improvement-plan.md` ("improvement plan") -- much of its section 3
  tunes the existing implementation and may not survive. Its 1.2, 1.3 and 1.4
  still hold.

## What the work is

Rebuild from **raw frames**. Existing `observation.h5` products are not a
trustworthy substrate: they carry an un-subtracted sky pedestal, stamps cut at a
fixed mean position with an arbitrary size, and further systematics added during
processing. Do not tune against them.

Three facts that cost real performance, and that any new implementation must
handle rather than inherit:

**1. Near-neighbour matching is probably the wrong tool.** The profile manifold
measures as roughly 4 components for 95% of the variance in a 180-pixel space.
If that holds on clean data, you want coverage of a low-dimensional manifold,
not the 100 nearest neighbours -- which is why the paper hit a curse of
dimensionality and needed negative coefficients. Provisional; re-measure.

**2. The published similarity metric partly measures brightness.** Profile
estimate variance scales as 1/N, so brighter references score better by being
less noisy. Measured: 71% of selected references are brighter than their target,
against 50% for a neutral metric. This one is analytic and does not depend on
the data.

**3. Stars drift, and it is reported as the dominant cause of lost
performance.** Rigid translation shared by all stars is mostly benign. The
damage is superpixel boundary crossings, field rotation (not common to all
stars, so it collapses the usable reference pool), and an aperture cut once at
the sequence mean. See improvement plan 3.2.

## The fleet is heterogeneous

Many units, many DSLR bodies: different black levels, white levels, gain, sensor
and pixel sizes. Same Bayer pattern, little else. Never hardcode a camera
constant -- measure it (saturation from the pixel histogram, gain by photon
transfer, pixel scale from the WCS, Bayer phase via `masks.infer_pattern`) or
resolve it from a profile keyed on the `CAMSN` serial, and **fail loudly rather
than defaulting**. Two bugs in this repo came from fleet-wide defaults.

No algorithm parameter in raw pixels unless it is about the detector grid:
express stamp sizes and apertures in multiples of measured FWHM, angular sizes
in arcsec. See improvement plan 1.4.

## Layout

- `src/panoptes/pipeline/lightcurve/` -- pure numpy, no I/O, no cloud, no
  notebook. `core` (the published steps plus a sky stopgap), `masks` (Bayer
  phase, apertures), `metrics` (scatter, binned scatter, red-noise beta, photon
  floor), `injection` (transit injection and recovery), `detection` (matched
  filter, blind scan, false-alarm threshold, completeness). New algorithm work
  belongs here.
- `src/panoptes/pipeline/` (the rest) -- orchestration and cloud I/O for the old
  pipeline. Requires Firestore, BigQuery, GCS.
- `notebooks/` -- the old execution path, via papermill. Legacy.
- `plans/`, `scripts/`, `tests/`.

## Running things

Python is run with `uv`; standalone scripts are PEP 723.

```bash
uv run scripts/benchmark_lightcurve.py OBS.h5 --channel r --sky-subtract --held-out
uv run scripts/survey_targets.py OBS.h5 --sky-subtract
uv run --no-project --with numpy --with scipy --with scikit-learn --with pytest \
  python -m pytest tests/
```

## Measuring a change

**The objective is detection, not scatter.** Score with
`lightcurve.detection.completeness` -- the fraction of injected transits
recovered above a false-alarm threshold, over a grid of depth and duration, with
the threshold set by `false_alarm_threshold` (circular time shifts, which
preserve red noise). A change that lowers RMS while lowering completeness is a
regression. See algorithm design 4.

Also required of any precision claim:

- Inject at the **pixel level**, before normalisation, so the whole pipeline
  sees the signal. Injecting into a finished lightcurve measures nothing.
- Score on **held-out frames**. Fitting many free parameters per target will
  absorb noise given the chance.
- Report the **per-target win rate** across a sample, not one target's median.
  A single flattering target is how the first baseline here came out 10x too
  optimistic.

Correcting a defect can make numbers look worse -- removing a bias toward flat
lightcurves is a real improvement presenting as a regression. Expect it.

## Conventions

- Cite a plan section by short name plus number: "algorithm design 2.1",
  "conformance audit 5.0". Never a bare number or a file path.
- `plans/` files are living documents. When an item is done, delete it -- no
  strikethrough, no "done" annotations. Git history is the record.
- Anything needing a human decision goes in improvement plan 6 as its own item.
- Selection masks are `True == included` everywhere in `lightcurve`. The
  `numpy.ma` convention is the opposite; convert only at that boundary.
- Stamp arrays unpack as `(height, width)`. Stamps are non-square in existing
  data; getting this backwards is conformance audit 5.4.
- `ruff check` and `ruff format`, line length 100.
