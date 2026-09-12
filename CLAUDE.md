# panoptes-pipeline

Differential photometry for PANOPTES: recovering exoplanet transits from DSLR
wide-field images with a Bayer colour filter array. Implements Gee et al.,
*On-sky Demonstration of Precision Photometry with Bayer Color Filter Arrays*.
Goal: 0.5% transit depths. Published result: ~1% in 30 min bins.

## Read these first

- `plans/conformance-audit.md` ("conformance audit") -- how far the code has
  drifted from the published algorithm, and every known defect.
- `plans/improvement-plan.md` ("improvement plan") -- the plan to reach 0.5%,
  with metrics, sequencing, and open decisions in improvement plan 6.

Read them on demand rather than assuming; they are the current state of play.
**improvement plan 5 is the sequencing** -- what to do first and why. Short
version: improvement plan 3.1 is one line and blocks every measurement in the
plan, so it comes before anything else; improvement plan 3.2 (drift) is the
largest expected gain and starts with a diagnostic, not a change.

## Three things to know before touching anything

**1. Stored stamps are not background-subtracted.** `ProcessFITS.ipynb` cell 16
computes `reduced_data = data - bg_data`, then cell 18 writes
`dict(raw=raw_data)` to `reduced_filename`. The subtracted array reaches only
`extras_filename`, gated on `save_extras`, default `False`. So every stamp in
every existing `observation.h5` carries bias plus sky -- typically 780 ADU per
pixel, which for a median source is over 95% of the stamp sum.

Consequences: any fractional RMS measured on those stamps is diluted and is not
photometric precision; transit depths are diluted 3-44x; and faint sources
appear to have *lower* scatter than bright ones because their stamp sum is a
stable pedestal. Fixing this is improvement plan 3.1 and blocks everything else.
`lightcurve.subtract_stamp_sky` is a labelled stopgap for already-processed
data, not the fix.

**2. The published algorithm's core step is missing from the notebooks.** Paper
sections 3.2.3 and 3.2.4 -- fitting per-reference coefficients in normalised
space, then applying them to the raw stamps -- are replaced in
`notebooks/working/MakeLightcurves.ipynb` by an unweighted mean of the top 100
references. `panoptes.pipeline.lightcurve` implements the real thing; the
notebooks have not been updated to use it.

**3. Stars drift, and that is reported as the dominant cause of lost
performance** across the archive, including the paper's own 2018-08-24 PAN012
sequence. The algorithm assumes a star holds its sub-pixel position; in practice
it rarely does. Rigid translation shared by all stars is mostly benign -- target
and references move together. The damage is from superpixel boundary crossings,
field rotation (which is *not* common to all stars and collapses the effective
reference pool), an aperture cut once at the sequence mean, and drift forcing an
18-pixel stamp that compounds problem 1. See improvement plan 3.2 before
proposing anything about reference selection.

## Layout

- `src/panoptes/pipeline/lightcurve/` -- the algorithm. Pure numpy, no I/O, no
  cloud, no notebook. `core` (paper 3.2.1-3.2.5 plus the sky stopgap), `masks`
  (Bayer phase, apertures), `metrics` (RMS, binned RMS, red-noise beta, photon
  floor), `injection` (transit injection and recovery). This is where new
  algorithm work belongs.
- `src/panoptes/pipeline/` (the rest) -- orchestration, cloud I/O, plate
  solving, catalog matching. Requires Firestore, BigQuery and GCS.
- `notebooks/` -- currently the production execution path, via papermill.
  Treat as legacy; improvement plan 4 moves the algorithm out of them.
- `plans/`, `scripts/`, `tests/`.

## Running things

Python is run with `uv`. Standalone scripts are PEP 723, declaring dependencies
in the inline metadata block:

```bash
uv run scripts/benchmark_lightcurve.py OBS.h5 --channel r --sky-subtract --held-out
uv run scripts/survey_targets.py OBS.h5 --channel r --sky-subtract
uv run --no-project --with numpy --with scipy --with scikit-learn --with pytest \
  python -m pytest tests/
```

`notebooks/PAN007_f6eb3d_20250930T030402/observation.h5` is an offline fixture:
3,234 sources, 42 frames, 10x18 stamps. No network or cloud credentials needed.

## Measuring a change

No precision claim without all of these:

- `--sky-subtract`, until improvement plan 3.1 lands. Numbers from
  un-subtracted stamps are not precision.
- `--held-out`. 100 free coefficients per target will fit noise given the
  chance; score only on frames the fit never saw.
- `survey_targets.py`, not `benchmark_lightcurve.py`. A single flattering
  target is how the first baseline in this repo came out 10x too optimistic.
  Report the per-target win rate, not just the median.
- Injection suppression alongside RMS. A change that lowers scatter while
  raising suppression has eaten signal, not removed noise.

Correcting a defect can make RMS look *worse* -- removing a bias toward flat
lightcurves is a real improvement presenting as a regression. Expect it.

## Conventions

- Cite a plan section by short name plus number: "conformance audit 5.0",
  "improvement plan 3.2". Never a bare section number or a file path.
- The `plans/` files are living documents. When an item is done, delete it --
  no strikethrough, no "done" annotations. Git history is the record.
- Anything needing a human decision goes in improvement plan 6 as its own item,
  not buried in prose.
- Selection masks are `True == included` everywhere in `lightcurve`. The
  `numpy.ma` convention is the opposite; convert only at that boundary.
- Stamp arrays unpack as `(height, width)`. Stamps are non-square in current
  data, and getting this backwards is conformance audit 5.4.
- `ruff check` and `ruff format`, line length 100.
