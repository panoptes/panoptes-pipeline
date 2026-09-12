# Improvement plan

Plan for taking the PANOPTES photometry algorithm from its published ~1%
(30 min binned) to the 0.5% needed for the transit survey.

Cite sections as "improvement plan 3.4". Companion document: the
[conformance audit](conformance-audit.md), which records how far the current
code has drifted from the published algorithm.

---

## 1. Goal and how it gets measured

### 1.1 Target

Recover 0.5% transit depths from DSLR wide-field photometry. The paper reports
2-4% unbinned and ~1% in 30 minute bins, on 8 < mV < 12 targets. Duty cycle has
since improved from ~12 to ~50 exposures per 30 minutes, so binning alone buys
roughly a factor of 2 -- but only for noise that actually averages down.

### 1.2 Metrics

Every change is scored on the same four numbers, produced by
`panoptes.pipeline.lightcurve.metrics`:

| Metric | Meaning | Why it matters |
|---|---|---|
| unbinned RMS | fractional scatter per exposure | the raw precision |
| 30 min binned RMS | scatter after binning | directly comparable to the paper |
| beta | red-noise factor, Pont et al. (2006) | correlated noise does *not* bin away |
| floor ratio | RMS / photon noise floor | how much headroom is left |

Plus, whenever a transit is injected: **recovered depth** and **suppression**.
A change that improves RMS while increasing suppression has not helped -- it
has partly eaten the signal. This pairing is what stops the work from
optimising into a flat line.

Beta is the metric that decides whether the 0.5% goal is reachable by binning.
If beta is materially above 1, more exposures will not get there and the
correlated term has to be attacked directly.

### 1.3 Benchmark datasets

1. **`PAN007_f6eb3d_20250930T030402`** -- already on disk at
   `notebooks/PAN007_f6eb3d_20250930T030402/observation.h5`. Runs offline with
   no cloud access. The fast regression fixture.
2. **The paper's HD 339461 sequence** (PAN012, 2018-08-24, 122 frames) -- the
   only dataset with a published number to reproduce. Needs the raw FITS
   recovered from GCS.
3. **Injection/recovery** -- synthetic transits injected at the pixel level
   into real frames. The only way to measure signal suppression.
4. **TESS-overlap targets** -- TOIs with published depths, for end-to-end
   truth. Slowest to assemble; do last.

Benchmark 1 and 3 work today. Benchmark 2 is an action item (6.1).

## 2. What is already in place

Built on branch `algorithm-v2`.

### 2.1 `panoptes.pipeline.lightcurve`

A pure-numpy implementation of paper sections 3.2.1-3.2.5, with no pandas, no
HDF5, no cloud client and no notebook. Four modules:

- `core` -- the algorithm. Normalisation, similarity scoring, reference
  selection, the **coefficient fit that was missing** (OLS, ridge, lasso,
  non-negative lasso, NNLS), comparison construction, differential photometry.
- `masks` -- Bayer color masks with explicit phase and pattern, circular and
  superpixel-grown apertures, optimal weights.
- `metrics` -- the 1.2 scorecard.
- `injection` -- box and trapezoid transit models, pixel-level injection,
  depth and suppression measurement.

41 tests, all passing. Conventions that previously caused silent bugs are now
explicit and enforced: selection masks are `True == included` throughout,
`select_references` refuses a pool that still contains the target
(conformance audit 5.3), and mask construction takes `(height, width)` in the
same order the data is ravelled (conformance audit 5.4).

### 2.2 The benchmark harness

`scripts/benchmark_lightcurve.py` (PEP 723, run with `uv run`) loads an
`observation.h5` and prints one table comparing algorithm variants on the same
target, including the current notebook behaviour as `legacy_mean`:

```
uv run scripts/benchmark_lightcurve.py notebooks/PAN007_.../observation.h5
uv run scripts/benchmark_lightcurve.py OBS.h5 --channel r --inject-depth 0.01
```

### 2.3 Baseline measurement

`PAN007_f6eb3d_20250930T030402`, PICID 265064120, 3,234 sources, 42 frames,
10x18 pixel stamps, 100 references:

| variant | full-stamp RMS | red-channel RMS |
|---|---|---|
| raw aperture | 0.38% | 1.14% |
| `legacy_mean` (what the notebook does today) | 0.18% | 0.98% |
| **fitted coefficients (paper 3.2.3-3.2.4)** | **0.15%** | **0.28%** |

Restoring the missing coefficient fit is worth a factor of **3.5 on the red
channel**, where the Bayer systematic is strongest and the mean-of-references
approach has least to work with. On the full stamp, where color effects
partially cancel, the gain is modest -- which is exactly why the notebook's
full-stamp-only photometry hid the problem.

Injection of a 1% transit over 0.4 h was recovered by every variant with
suppression below 5%, so flux marginalisation is doing its job.

**These numbers are one target in one observation and are not yet a baseline.**
The sequence is 42 frames over ~50 minutes, so the 30 min and beta columns have
too few bins to mean anything. Establishing a real baseline is action item 6.2.

## 3. Precision work, in priority order

### 3.1 Restore the coefficient fit

Already implemented in `lightcurve.core`. What remains is putting it on the
production path, which is section 4.

**Expected gain: large.** Measured at 3.5x on one channel of one target.

### 3.2 Choose and tune the regulariser

Open question, not a porting job. The paper does not state its regulariser; the
46-of-100 sparsity in Figure 7 implies L1, but the strength is unknown.

Candidates, all implemented: ridge (dense, stable), lasso (sparse, matches the
figure), non-negative lasso (sparse and physically interpretable -- a negative
coefficient means extrapolating outside the reference set, which the paper
observes and defends but which also invites overfitting), NNLS.

Method: sweep `alpha` across a decade grid for a few hundred targets, scoring
on 1.2 metrics *and* suppression. Expect an interior optimum -- too little
regularisation overfits 100 free parameters to noise, too much collapses toward
the unweighted mean. On the early data the lasso settings tried so far keep
only ~10 references and score *worse* than OLS, so the strength is currently
wrong, not the method.

**Expected gain: moderate.** Also the cheapest experiment available.

### 3.3 Per-channel selection and fitting

The paper selects references and fits coefficients using all pixels, then
splits color only at the final photometry. But the systematic being corrected
*is* the color-dependent pixel response, so the natural extension is to select
and fit each channel independently. `make_lightcurve(select_on_channel=True)`
already supports this.

Against: each channel has a quarter (red, blue) or half (green) of the pixels,
so the fit has proportionally less data constraining the same 100 coefficients,
and may overfit. Test it, do not assume it.

**Expected gain: moderate, and interacts with 3.2.**

### 3.4 Apertures

Two changes:

1. **Grow every aperture to whole superpixels.** A hard-edged aperture clips
   superpixels differently as the star drifts; measured, the green:red pixel
   ratio inside a radius-2 aperture swings from 0.5 to 4.5 across half-pixel
   centroid shifts (conformance audit 5.8). That is a color-dependent flux
   modulation at the tracking period -- precisely the systematic the algorithm
   exists to remove, reintroduced at the last step.
2. **Then try optimal (inverse-variance) weighting** using the synthetic
   comparison star as the PSF profile. The comparison is already a
   noise-suppressed model of the target's own pixel distribution, which makes
   it a better weight map than any analytic profile.

Both are in `lightcurve.masks`. Paper section 6.1 lists the aperture as future
work.

**Expected gain: moderate to large, especially per-channel.**

### 3.5 Flat fields and local background

Paper section 6.1 lists both. Flat-fielding removes the pixel-to-pixel
sensitivity variation that currently has to be absorbed by the reference
matching, and would let background subtraction work properly in the vignetted
corners -- where the paper's own example target sat.

The paper deliberately avoided calibration, on the reasoning that it could
alter systematics in unknown ways. That reasoning deserves testing rather than
inheriting: with the harness in 2.2, "does flat-fielding help or hurt" is now a
measurable question rather than a judgement call.

Needs: a flat-field acquisition procedure for the fleet, or sky flats built by
stacking. This is the longest-lead item and the one with the largest
uncertainty.

**Expected gain: unknown, potentially large. Highest effort.**

### 3.6 Signal-safe reference selection

References are currently chosen by similarity across *all* frames, in-transit
frames included. Flux marginalisation (Eq. 1) protects against most
self-subtraction, and the measured suppression below 5% supports that -- but
that was one injection at one depth.

Work: sweep injected depth and duration against suppression, to map where the
protection breaks down. Add out-of-transit-only reference selection via
`frame_weights` (already supported) and measure whether it costs precision.

**Expected gain: not precision, but it bounds a bias that would otherwise
contaminate every depth measurement the survey produces.**

### 3.7 Per-point uncertainties

There are none today. Every lightcurve is a bare array of relative fluxes with
no error bars, so no transit fit downstream can be weighted or assessed. Needs
propagation of photon, read and background noise through the coefficient fit
into the final ratio.

**Expected gain: no RMS change, but nothing downstream is trustworthy without
it.**

### 3.8 Reference pool scale and search cost

Similarity search is O(p^2) (conformance audit 5.16). Paper section 3.2.2
suggests clustering. Options: PCA on normalised stamps then approximate nearest
neighbours, or KD-tree in a reduced feature space.

This is a throughput problem, not a precision problem -- but it becomes a
precision problem the moment it is cheap enough to raise the reference pool
well above 100, which 3.2 may want.

**Expected gain: throughput; enables larger pools.**

### 3.9 Frame and pixel quality weighting

`frame_weights` is plumbed through but unused. Candidates: down-weight frames
by measured FWHM, background level or airmass; mask individual hot pixels and
cosmic rays rather than discarding whole frames.

**Expected gain: small, but cheap.**

## 4. Making it runnable

Target architecture: **offline-first library plus CLI**. The algorithm must run
on a local directory of FITS files with no Google Cloud project, because a
citizen-science project cannot require every contributor to hold cloud
credentials, and because an algorithm that cannot run in a test cannot be
improved with confidence.

### 4.1 Package layout

`panoptes.pipeline.lightcurve` (done, 2.1) is the pattern: pure functions over
arrays, no I/O. Extend it to the image-level steps currently trapped in
`ProcessFITS.ipynb` -- background subtraction, source detection, catalog
matching -- so each is importable and testable.

### 4.2 CLI

Extend the existing Typer app so `panoptes-pipeline` can process a local
directory end to end: calibrate, detect, match, extract stamps, build
lightcurves, write output. Today `cli/main.py` only orchestrates papermill and
points at two notebook paths that do not exist (conformance audit 5.14).

### 4.3 Storage adapters

Firestore, BigQuery and GCS move behind interfaces with local implementations:
catalog from a local parquet file (already half-supported via
`CatalogSettings.catalog_filename`), metadata to local JSON, results to local
parquet. Cloud becomes a deployment choice rather than a hard requirement.

### 4.4 Notebooks become demos

Once the library owns the algorithm, the notebooks import it and plot. They
stop being the implementation. `notebooks/working/` -- 20 files including six
`RunProcessFits-Copy*.ipynb` and two `Untitled` -- gets archived or deleted.

### 4.5 CI

`.github/` is empty. Add: ruff, pytest, and a smoke run of
`scripts/benchmark_lightcurve.py` against a trimmed fixture, so a regression in
the numbers fails the build rather than being discovered months later.

## 5. Sequencing

**Now** -- 3.1 on the production path, 3.2 alpha sweep, 3.4 superpixel
apertures. All three are implemented or nearly so, and all three are measurable
against benchmark 1 today.

**Next** -- 3.3 per-channel fitting, 3.6 suppression mapping, 3.7 uncertainties,
4.1-4.3 library and CLI.

**After** -- 3.5 flat fields, 3.8 search scaling, 3.9 quality weighting,
benchmark 4.

Deliberately deferred: anything that improves throughput before precision is
established, and any move to fit transit parameters. Get one target right
first.

## 6. Action items

These need a decision or something only you can provide.

**6.1 -- Recover the paper's HD 339461 data.** Benchmark 2 is the only dataset
with a published number to reproduce, and it is the only way to tell a real
improvement from a better observation. Are the 122 raw frames from PAN012
2018-08-24 still in GCS, and can this session reach them?

**6.2 -- Confirm the baseline scope.** Proposal: run the harness over ~200
targets spanning 8 < mV < 12 in the PAN007 sequence and freeze the result as
the reference baseline. Needs your sign-off on the magnitude range and target
count before it becomes the number everything is measured against.

**6.3 -- Decide on the regulariser experiment scope (3.2).** Sweeping alpha x
method x reference count over hundreds of targets is the single most
informative experiment available, and the most compute. Should it run locally,
or is Cloud Run still a live deployment target?

**6.4 -- Flat fields (3.5).** Does any PANOPTES unit currently take flats, or
would this need a new acquisition procedure in POCS? This gates the
longest-lead item in the plan.

**6.5 -- Confirm the camera fleet constants.** `settings.py` hardcodes
`zero_bias=512` and `saturation=15872` for every camera, while the paper's
EOS 100D is 2048 and 11535 (conformance audit 4.5). Which cameras are in the
fleet now, and where should per-unit constants live?

**6.6 -- Branch disposition.** `algorithm-v2` is cut from `pipeline-working`,
which has uncommitted changes and untracked deployment files. Confirm that is
the right base, and whether the rebuilt pipeline should eventually land on
`develop` or on a fresh `main`.

## 7. Risks

**The 0.5% goal may not be reachable from a single camera.** The paper states
its result "approaches the fundamental noise floor possible from a single
camera". If the floor ratio in 1.2 comes back near 1, the remaining paths are
combining units or longer exposures, not algorithm work. Measuring the floor
ratio early (6.2) tells us which problem we are solving, and should happen
before any large effort is spent on 3.5.

**Overfitting is the standing hazard.** 100 free coefficients fit to one
target's pixel history will happily absorb a transit. Suppression (1.2) is
reported on every run for this reason, and no precision result should be
accepted without it.

**Fixing correctness can make the numbers look worse.** Correcting the target
self-reference defect (conformance audit 5.3) removes a bias *toward* flat
lightcurves, so measured RMS may rise. That is a real improvement presenting as
a regression, and it needs to be noted when it happens rather than debugged.
