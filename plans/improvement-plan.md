# Improvement plan

Plan for taking the PANOPTES photometry algorithm from its published ~1%
(30 min binned) to the 0.5% needed for the transit survey.

Cite sections as "improvement plan 3.6". Companion document: the
[conformance audit](conformance-audit.md), which records how far the current
code has drifted from the published algorithm.

This document is the *reasoning*. What is planned, in progress and done is in
the [issue tracker](https://github.com/panoptes/panoptes-pipeline/issues), and
anything needing a human decision is an issue with the `decision` label. This
file used to carry an action-item list as well; two half-maintained trackers is
one too many.

---

## 1. Goal, constraints and how success is measured

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
optimizing into a flat line.

Beta is the metric that decides whether the 0.5% goal is reachable by binning.
If beta is materially above 1, more exposures will not get there and the
correlated term has to be attacked directly.

### 1.3 Benchmark datasets

Roughly ten years of raw PANOPTES data is available on the project's processing
server -- 12,439 sequences and 563,566 frames, discoverable through
[`panoptes-data`](https://github.com/panoptes/panoptes-data). That repository is
ours and it is downstream of this one, so if selection needs a query the client
cannot express, the fix is to add it there rather than to reimplement archive
logic here.

**Selection runs in two passes, and the reason is not convenience.** Cut on
header facts first -- frame count, duration, unit, camera uid, ISO, exposure,
moon, airmass, field -- then download and measure drift and seeing locally
before cutting again. The archive's FWHM, source counts and solved positions are
outputs of the implementation being replaced, so ranking candidates by them to
build the substrate for judging new code is circular. Duration is also not
expressible against the current index at all, and `num_images` counts frames
uploaded rather than usable. See data contract 9.

Benchmark selection is therefore a choice rather than a constraint, and **the
starting point is raw frames, not existing `observation.h5` products**. Those
carry an un-subtracted sky pedestal (conformance audit 5.0), stamps cut at a
fixed mean position, and further systematics introduced during processing. 3.1
changes the image-level reduction in any case, so a fixture built from them
would be contaminated from the start. See algorithm design 5 for which of the
measurements taken from one of those files survive and which do not.

Record measured drift alongside every selected sequence -- per 3.2 it is
probably the strongest predictor of achievable precision, and selecting on it
deliberately beats discovering it afterwards.

Currently on disk and usable offline:
`notebooks/PAN007_f6eb3d_20250930T030402/observation.h5` -- 3,234 sources, but
only 42 frames over ~50 minutes. Enough to regression-test code, far too short
to measure binned RMS or beta. It is a fixture, not a benchmark.

What to select, in priority order. Each entry earns its place by answering a
question nothing else can.

**A. The paper's own sequence.** PAN012, 2018-08-24, HD 339461, 122 x 35 s.
The only dataset with a published number to reproduce (2-4% unbinned, ~1%
binned). Without it there is no way to separate a better algorithm from a
better night.

**B. A long, high-duty-cycle sequence from a modern unit.** At least 300 frames
over 3+ hours. The paper's duty cycle was ~12 exposures per 30 minutes; current
units manage ~50. This is the largest gap in what we have -- beta and the 30 min
column are meaningless below roughly 100 frames, and beta is the metric that
decides whether 0.5% is reachable by binning at all (1.2).

**C. A dark-sky / bright-moon pair on the same field.** The paper's data was
taken under a 95% moon from an urban site, and conformance audit 5.0 shows the
sky pedestal dominates the stamp sum. A matched pair separates how much of the
systematic is sky and how much is detector. Cheap to select, high information.

**D. A known transit with published depth.** A TOI or known hot Jupiter with
full ingress and egress. Ground truth for depth recovery, and the only check on
suppression that does not rely on injection.

**E. Crowded and sparse fields.** The algorithm's premise is that a large
number of sources per frame makes a good reference pool. Low and high galactic
latitude sequences bound how that degrades.

**F. Two cameras on the same field, same night**, ideally two *different*
camera models. Separates detector systematics from atmospheric ones, and
directly addresses the risk in 7 that the floor is per-camera -- if two units'
residuals are uncorrelated, combining units is the path to 0.5%; if they track
each other, it is not. Different models also give the first read on how much of
the result is body-specific.

**H. A spread in PSF sampling.** Per 1.4, the strength of the Bayer systematic
is governed by FWHM in pixels, so the benchmark set should span the fleet's
range rather than clustering at one pixel scale. Otherwise every parameter
tuned here is tuned for one sampling regime.

**G. A deliberately poor night.** High airmass, thin cloud, or bad tracking.
Frame rejection and quality weighting cannot be tuned on good data.

Plus, not an observation: **raw frames from any long sequence** are what the
frame-scale background work in 3.6 needs, and a median stack of one gives a sky
flat with no new acquisition procedure.

Start with A and B. They unblock the baseline; the rest can follow.

**Synthetic injection/recovery** remains the fourth leg and works today, on
whatever data is loaded.

### 1.4 The fleet is heterogeneous

The archive is not one instrument. It is many units, many different DSLR
bodies, with different black levels, different white levels, different gain,
and different sensor and pixel sizes. The Bayer pattern is common; almost
nothing else is. Everything built here has to be generic across that, and the
current code is not: `settings.py` hardcodes one bias, one saturation and one
gain for every camera in the fleet (conformance audit 4.5).

**Why this bites harder here than in a typical pipeline.** The algorithm's
entire subject is the interaction between the PSF and the 2x2 superpixel, and
the dimensionless number that governs it is the PSF width *in pixels*. A
well-sampled PSF spreads across many superpixels and the color-sampling
systematic partly averages itself out; an undersampled one, which is the
paper's regime, is where it bites hardest. Pixel scale follows from pixel size
and focal length, so **the size of the effect this algorithm exists to remove
varies across the fleet**. Stamp size, aperture radius and the useful number of
references are all likely to differ per unit, and so is the expected gain.

**Two sharp hazards, and one that is milder than it looks.**

*Saturation is the dangerous one.* A fleet-wide threshold either fails to mask
saturated pixels on a camera with a lower white level -- silently corrupting the
brightest and otherwise best targets -- or masks good pixels on a camera with a
higher one, discarding those same targets. Neither failure is loud. White level
also moves with ISO and bit depth.

*Gain is the other.* It sets the photon noise floor in 1.2, and the floor ratio
is how we decide whether 0.5% is reachable at all rather than being a
per-camera limit (7). A wrong gain makes that judgment meaningless.

*Black level matters less than it first appears* -- once 3.1 lands.
`Background2D` estimates bias and sky jointly and removes both, so the exact
bias does not enter Eq. 1 directly. It is still needed for the saturation
threshold and the noise model, so it is not optional, just narrower in scope
than it looks.

**Measure, do not configure.** A hand-maintained table of camera constants will
be wrong for the first body nobody registered, and wrong silently. Most of what
is needed is recoverable from the data:

| Quantity | Where it comes from |
|---|---|
| pixel scale | the WCS, per frame, free |
| image dimensions | the array shape (already done) |
| saturation / white level | `WHTLVLN`/`WHTLVLS` in the header, where POCS still wrote them; otherwise the pixel histogram -- there is a hard cutoff; take the recurring top value across many frames, per camera and ISO |
| gain, read noise | photon transfer: variance against mean across frame pairs. Ten years of multi-frame sequences makes this straightforward per camera and ISO |
| PSF FWHM | already measured per frame by photutils |
| Bayer phase | `MEASRGGB` is parsed out of the header in `extract_metadata`, but current POCS no longer writes it (data contract 2.2), so `masks.infer_pattern` is the durable answer |
| black level | header where present, otherwise measured once per camera and stored as data |

**Resolution order, and no silent defaults.** Key a `CameraProfile` on the
camera **uid** -- the six-character identifier in `INSTRUME` and in every
`sequence_id`, derived from the camera's own serial. Not `CAMSN`, which current
POCS no longer writes and which is missing from a sixth of recent records, and
never the `Cam00` slot name, which is assigned regardless of which body is
installed. See data contract 2.3.

Resolve: header, then measured from this observation, then the stored profile
for that uid, then **fail loudly**. Fields should have no fleet-wide default at
all, so a missing value raises instead of quietly producing a wrong number.
`zero_bias: float = 512` applying to every camera ever built is the exact
anti-pattern to remove -- as is `effective_gain: float = 1.5`, which is a POCS
class constant this repository copied and which is best fixed there (data
contract 6.3).

**No parameter in raw pixels** unless it is genuinely about the detector grid,
such as superpixel alignment. Convert stamp size and aperture radius to
multiples of the measured FWHM (then round up to whole superpixels); the
background box from 79x84 pixels to an angular size or a fraction of the frame;
drift tolerances to arcseconds, since mount error is angular. Already correct:
`max_separation_arcsec`, and a detection threshold in sigma.

**Cross-unit combination happens at the lightcurve level, never the pixel
level.** Two cameras do not share a pixel scale, a stamp size or a PSF, so a
reference pool cannot span units. If combining units is the path to 0.5% (7),
it combines normalized lightcurves after the fact.

**The heterogeneity is also an asset.** The fleet is a natural experiment in
pixel scale. Measuring how the achievable precision varies with PSF sampling
tells us what to specify for future units, which is a result the project wants
independently of this algorithm.

## 2. What is already in place

Built on branch `algorithm-v2`.

### 2.1 `panoptes.pipeline.lightcurve`

A pure-numpy implementation of paper sections 3.2.1-3.2.5, with no pandas, no
HDF5, no cloud client and no notebook. Four modules:

- `core` -- the algorithm. Normalization, similarity scoring, reference
  selection, the **coefficient fit that was missing** (OLS, ridge, lasso,
  non-negative lasso, NNLS), comparison construction, differential photometry.
- `masks` -- Bayer color masks with explicit phase and pattern, circular and
  superpixel-grown apertures, optimal weights.
- `metrics` -- the 1.2 scorecard.
- `injection` -- box and trapezoid transit models, pixel-level injection,
  depth and suppression measurement.

`core` also carries `subtract_stamp_sky`, the stopgap for 3.1.

Conventions that previously caused silent bugs are now explicit and enforced:
selection masks are `True == included` throughout, `select_references` refuses
a pool that still contains the target (conformance audit 5.3), and mask
construction takes `(height, width)` in the same order the data is raveled
(conformance audit 5.4).

### 2.2 The benchmark harness

Two scripts, both PEP 723, both offline.

`scripts/benchmark_lightcurve.py` answers "what happened to this star": one
table comparing variants on one target, including today's notebook behavior as
`legacy_mean` and a fair ensemble baseline as `ensemble_scaled`.

`scripts/survey_targets.py` answers "what happens in general": the distribution
of scatter across many targets and the fraction each variant actually wins on.
A median improvement that only holds for half the sample is not an improvement.

```
uv run scripts/benchmark_lightcurve.py OBS.h5 --channel r --sky-subtract --held-out
uv run scripts/survey_targets.py OBS.h5 --channel r --sky-subtract
```

`--held-out` selects references and fits coefficients on alternate frames, then
scores only on frames the fit never saw. `--sky-subtract` applies the 3.1
stopgap. Both default off in `benchmark_lightcurve.py`, which prints a loud
warning when the sky pedestal is left in.

### 2.3 Baseline measurement

All numbers below: `PAN007_f6eb3d_20250930T030402`, 80 brightest unsaturated
targets, one color channel, 100 references, scored on **held-out frames** (references
selected and coefficients fitted on alternate frames only, then scatter
measured on the frames the fit never saw).

**On stamps as stored** -- which is to say, with the sky pedestal the pipeline
failed to subtract (conformance audit 5.0):

| variant | median RMS | beats ensemble |
|---|---|---|
| raw aperture | 1.45% | -- |
| `ensemble_scaled` (flux-scaled mean of references) | 1.29% | -- |
| fitted coefficients (paper 3.2.3-3.2.4) | 0.63% | 98% of targets, 2.15x |

**With the sky pedestal removed** (`--sky-subtract`):

| variant | median RMS | beats ensemble |
|---|---|---|
| raw aperture | 6.56% | -- |
| `ensemble_scaled` | 5.99% | -- |
| fitted coefficients | 4.80% | 72% of targets, **1.15x** |

Three things to take from this.

**Most of the apparent gain was the background.** On un-subtracted stamps the
coefficient fit looks like a 2x improvement. It is not fitting stellar
morphology there -- it is fitting smooth background structure shared between
neighboring stars, which it does very well. Remove the pedestal and the honest
gain is 1.15x, winning on roughly three targets in four. Real, worth having,
and an order of magnitude less than the first measurement suggested.

**The true precision is nowhere near the published result.** 4.8% on the red
channel against the paper's 2-4%. Part of that is the stopgap sky subtraction
used here -- a per-frame median of the stamp's outer pixels, which over-subtracts
stellar wings on a 10x18 stamp and inflates every row of the second table. The
relative comparison between variants is trustworthy because every variant sees
the same data; the absolute numbers are pessimistic by an unknown factor. They
stop being a guess once 3.1 is done.

**Which color channel, exactly, is not yet established.** These were taken
with a hardcoded RGGB assumption. Measured afterwards from the sky levels of
star-free stamps, the two greens in this data sit on the *diagonal*, so the
pattern is GRBG or GBRG and the subset labeled "red" was in fact one of the
greens. Every variant used the same pixel subset, so the comparison between
them stands; the channel name does not. `masks.infer_pattern` now derives it
from the data and refuses to guess red from blue without the `MEASRGGB` header.
Re-label these once a sequence is reduced with the header available.

**It is not overfitting.** Held-out scatter matches in-sample scatter to within
3% for both OLS and ridge, so 100 free coefficients per target are not
absorbing noise. That was the main risk and it is cleared.

Caveats: one observation, 42 frames over ~50 minutes, 21 of them held out. The
30 min and beta columns have too few bins to mean anything yet. Establishing a
real baseline needs a decision on scope first (issue #136).

## 3. Precision work, in priority order

> **Contingent on architecture.** 3.3 through 3.5 tune the published
> implementation -- the coefficient fit, its regularizer, per-channel variants.
> If the manifold model in algorithm design 3 wins the head-to-head, those do
> not survive in their current form. 3.1, 3.2, 3.6 and 3.7 apply to any
> implementation. Do not work through this list top to bottom before that test.

### 3.1 Actually subtract the background

Nothing else on this list can be measured honestly until this is fixed. The
defect was one line -- `ProcessFITS.ipynb` cell 18 wrote `raw_data` to
`reduced_filename` rather than `reduced_data` (conformance audit 5.0) -- and
that notebook has since been deleted, so this is now a requirement on the
replacement rather than a patch: whatever writes the reduced image writes the
background-subtracted one. Then reprocess at least one observation so there is
an uncontaminated fixture.

Alongside it, restore the paper's 11x12 median filter (conformance audit 4.1),
and keep the background map as a saved product so per-stamp local background
becomes possible later (3.7).

`lightcurve.subtract_stamp_sky` exists as a stopgap for observations already
processed, and the benchmark scripts take `--sky-subtract`. It estimates sky
from the stamp's outer pixels, which over-subtracts stellar wings. It is a way
to keep measuring, not a fix.

**Expected gain: this does not improve precision, it reveals it.** Every number
in 2.3 and every transit depth the survey produces is diluted until it lands.

### 3.2 Drift and sub-pixel sampling

The algorithm assumes a star holds its position, so that its sub-pixel phase on
the Bayer array is a fixed hidden parameter to be matched. In practice, across
the archive and including the paper's own 2018-08-24 PAN012 sequence, stars
drift -- and this is reported as the dominant cause of lost performance.

**What drift does and does not break.** A pure rigid translation shared by every
star is largely benign: star *k* sits at phase `φ_k + d_i`, the drift `d_i` is
common, so target and references move together and Eq. 2 still matches what it
is supposed to match. That is the case the paper tolerates when it claims
robustness to multi-pixel tracking error. The damage comes from four other
places:

- **Superpixel boundary crossings.** A star whose phase crosses a boundary
  changes which color samples its core, discretely. Whether it crosses depends
  on where it started, so the population splits and the "locally linear"
  assumption in paper section 3.1 fails across the split.
- **Field rotation.** With imperfect polar alignment the motion is not common
  to all stars: it depends on field position. The effective reference pool
  collapses to stars at similar radius and azimuth from the rotation center,
  which may be a small fraction of the 3,000 available.
- **A fixed aperture cut at the mean position.** With drift the aperture samples
  a varying and color-dependent fraction of the PSF -- the same 0.5 to 4.5
  green:red swing measured in conformance audit 5.8, now modulated at the
  tracking period.
- **Stamp inflation.** Large drift pushes the stamp from 10 to 18 pixels per
  axis, and given conformance audit 5.0 that means proportionally more
  background and less star in every sum.

Intra-exposure drift -- trailing -- is a genuine PSF change and is not
recoverable. Everything else is.

**Recommendations, in order.**

1. **Re-center stamps per frame on the nearest superpixel.** Cut each frame's
   stamp at that frame's position rounded to a whole superpixel, rather than
   once at the sequence mean. Whole-superpixel shifts preserve Bayer phase
   exactly, so this decouples bulk drift (absorbed by re-centering) from
   sub-pixel phase (what the algorithm is actually designed to match). It keeps
   a 10x10 stamp usable under multi-pixel drift, which also cuts the background
   dilution. Cost: the stamp then covers different detector pixels over time, so
   flat-field variation becomes a time-varying term -- test against a sky flat
   (3.7) rather than assuming it is free.

2. **Center the aperture per frame**, grown to whole superpixels (3.6). A fixed
   aperture under drift is wrong in a color-dependent way on most frames.

3. **Use the positions we already measure.** `catalog_wcs_x/y` is stored per
   star per frame, so each star's sub-pixel phase is known, not hidden. Paper
   section 3.1 deliberately chose an empirical morphology search over solving
   for parameters like sub-pixel position. Drift is a good reason to revisit
   that: condition reference selection on phase directly, or add it as a
   feature, instead of hoping the search rediscovers it.

4. **Treat this as the intrapixel problem it is.** Sub-pixel position modulating
   measured flux is the Spitzer intrapixel effect, and the standard solution is
   pixel-level decorrelation (Deming et al. 2015). PLD normalizes each frame's
   pixels by their sum -- *precisely* paper Eq. 1 -- and then uses those
   normalized pixels as regressors against the lightcurve. PANOPTES computes the
   same quantity and uses it only for star-to-star matching. Adding the target's
   own normalized pixels as regressors is a small change to the existing design
   and is the technique built for this failure mode. It carries a known risk of
   absorbing the transit, which is what `frame_weights` and the suppression
   metric already exist to bound.

   This also inverts the framing. A star pinned to one sub-pixel gives no
   leverage to measure the intrapixel response; a star that drifts *samples* it.
   Ten years of drifting data is a training set for a sensitivity map rather
   than a loss.

5. **Time-local coefficients.** Eq. 4 fits one coefficient vector for the whole
   sequence. If drift changes the character of the systematic across the
   observation, fit in a sliding window instead. More free parameters, so pair
   it with held-out scoring and injection before believing any gain.

6. **Diagnose rotation versus translation** before any of the above. Fit the
   per-star offsets to translation plus rotation about a free center and report
   the split. The deleted `ProcessObservation.ipynb` filtered frames on the
   *mean* xy offset across all stars (cell 44), which assumes pure translation
   and hides rotation entirely. Do not reproduce that.

7. **Down-weight trailed frames.** `photutils_eccentricity` and
   `photutils_fwhm` are already computed per source. An elongated frame has a
   different morphology, not a different position, and should be weighted down
   rather than matched.

**Measure before building.** Regress each target's residual flux against its
measured `(Δx, Δy)` and the quadratic terms. The fraction of variance explained
says how much is recoverable here and ranks items 1-5 on real data instead of
argument.

**Expected gain: potentially the largest item in this plan**, on the direct
evidence that drift is what cost most of the performance.

**Consequence for dataset selection (1.3).** Prioritize sequences spanning a
range of drift behavior, and record measured drift alongside each. Dataset A
is then valuable as a hard case to beat rather than a number to match -- if its
drift is severe, reproducing 2-4% on it is the wrong target and beating it is
the right one.

### 3.3 Restore the coefficient fit

Already implemented in `lightcurve.core`. What remains is putting it on the
production path, which is section 4.

**Expected gain: 1.15x on the single color channel, winning on ~72% of targets**, once
the background is subtracted. Modest but real, and it is the step that makes
the rest of section 3 worth doing -- an unweighted mean has no parameters to
improve.

### 3.4 Choose and tune the regularizer

Open question, not a porting job. The paper does not state its regularizer; the
46-of-100 sparsity in Figure 7 implies L1, but the strength is unknown.

Candidates, all implemented: ridge (dense, stable), lasso (sparse, matches the
figure), non-negative lasso (sparse and physically interpretable -- a negative
coefficient means extrapolating outside the reference set, which the paper
observes and defends but which also invites overfitting), NNLS.

Method: sweep `alpha` across a decade grid for a few hundred targets, scoring
on 1.2 metrics *and* suppression. Expect an interior optimum -- too little
regularization overfits 100 free parameters to noise, too much collapses toward
the unweighted mean. On the early data the lasso settings tried so far keep
only ~10 references and score *worse* than OLS, so the strength is currently
wrong, not the method.

**Expected gain: moderate.** Also the cheapest experiment available.

### 3.5 Per-channel selection and fitting

The paper selects references and fits coefficients using all pixels, then
splits color only at the final photometry. But the systematic being corrected
*is* the color-dependent pixel response, so the natural extension is to select
and fit each channel independently. `make_lightcurve(select_on_channel=True)`
already supports this.

Against: each channel has a quarter (red, blue) or half (green) of the pixels,
so the fit has proportionally less data constraining the same 100 coefficients,
and may overfit. Test it, do not assume it.

**Expected gain: moderate, and interacts with 3.4.**

### 3.6 Apertures

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

### 3.7 Flat fields and local background

**Tested and rejected at stamp scale.** The obvious follow-on from conformance
audit 5.0 is that if the coefficient fit models background that well, it should
*be* the background model: build a basis from star-free stamps, fit it to each
target's sky annulus, evaluate over the core, subtract. Measured on
`PAN007_f6eb3d_20250930T030402`, fitting on half the annulus superpixels and
scoring on the other half:

| sky model | held-out residual | vs constant |
|---|---|---|
| per-color constant | 65 ADU | 1.00x |
| per-color plane | -- | worse |
| 2-10 star-free PCA modes | 84-125 ADU | 0.52-0.78x |

Every fitted model is worse than a per-color constant, and downstream single-channel
photometry degrades from 4.65% to 89%.

The singular values of the star-free stamp population say why: `[1, 0.026,
0.023, 0.019, 0.013, 0.012]`. One dominant mode -- the overall level -- then a
flat noise tail. A 10x18 stamp is about a fifth of one `Background2D` box, and
across it the sky really is flat to within the per-pixel noise. There is no
structure for a richer model to capture, so extra modes fit noise in an
18-pixel red fit region and then extrapolate into the core.

Two things follow. A per-color constant is close to optimal *at stamp scale*,
so the 3.1 stopgap is not leaving much on the table. And the apparent quality of
the coefficient fit on un-subtracted stamps was never rich spatial modeling --
it was tracking one scalar per frame per color, the sky level, which is almost
perfectly correlated between neighboring stars. That makes the 2.15x in 2.3
less mysterious and more clearly an artifact.

**Where the idea should go instead: frame scale.** Across the full 6000x4000
frame there is real structure -- vignetting, gradients, scattered light, and
per-pixel sensitivity. A basis built from star-free regions of the whole frame,
or PCA over many frames' background maps, is a well-founded replacement for the
`Background2D` mesh and is untested here only because the raw frames were not
available in this session. That is the experiment worth running, not the
stamp-scale one.

The flat field is the part a background fit cannot reach, because it is
multiplicative. Each target's stamp sits at a fixed detector position, so the
sensitivity pattern beneath it is constant in time -- which means a median stack
over many frames yields a sky flat without any new acquisition procedure. Worth
trying before asking the fleet to take dome flats (issue #144).



Paper section 6.1 lists both. Flat-fielding removes the pixel-to-pixel
sensitivity variation that currently has to be absorbed by the reference
matching, and would let background subtraction work properly in the vignetted
corners -- where the paper's own example target sat.

The paper deliberately avoided calibration, on the reasoning that it could
alter systematics in unknown ways. That reasoning deserves testing rather than
inheriting: with the harness in 2.2, "does flat-fielding help or hurt" is now a
measurable question rather than a judgment call.

Needs: a flat-field acquisition procedure for the fleet, or sky flats built by
stacking. This is the longest-lead item and the one with the largest
uncertainty.

**Expected gain: unknown, potentially large. Highest effort.**

## 4. Making it runnable

Target architecture: **offline-first library plus CLI**. The algorithm must run
on a local directory of FITS files with no Google Cloud project, because a
citizen-science project cannot require every contributor to hold cloud
credentials, and because an algorithm that cannot run in a test cannot be
improved with confidence.

### 4.1 Package layout

`panoptes.pipeline.lightcurve` (done, 2.1) is the pattern: pure functions over
arrays, no I/O. Extend it to the image-level steps -- background subtraction,
source detection, catalog matching -- so each is importable and testable. They
were last implemented in `ProcessFITS.ipynb`, which is deleted; git history is
the reference for what those cells did.

**Keep `images.py` where it earns its place; rebuild from stamp extraction
down.** Settled. Its wrappers around astropy, photutils and astrometry.net for
bias, background, plate solving, detection and catalog matching are ordinary
working code, and rewriting them would re-derive a solved problem while adding
new bugs. What does not survive is stamp extraction -- per-frame superpixel
re-centering (3.2) makes it a different operation, not a tweaked one -- and
everything downstream of it.

The boundary is therefore: image in, matched source catalog out, reused. Stamps
and everything after, rebuilt.

### 4.2 CLI

A Typer app so `panoptes-pipeline` can process a local directory end to end:
calibrate, detect, match, extract stamps, build lightcurves, write output.

There is nothing left to extend. The previous `cli/main.py` was papermill
orchestration pointing at two notebook paths that did not exist (conformance
audit 5.14), and it was deleted along with the notebooks, so the console script
is currently absent rather than broken. It comes back when there is a library
for it to drive.

### 4.3 No cloud path

**Settled: outright removal, not adapters.** Firestore, BigQuery and GCS are
gone from this package -- `utils/gcp/` deleted, the `cloud` extra deleted, no
Google client anywhere in the dependency tree. Every module imports from a plain
`uv sync`.

The catalog is a local file named by `CatalogSettings.catalog_filename`, and
`sources.get_stars` raises rather than falling back to a network lookup when it
is unset. Metadata and results are local files.

**Catalog format is parquet, ECSV or CSV, chosen by suffix.** Parquet is the
better default for an all-sky catalog: at the 6 < mV < 13 the settings ask for,
that is millions of rows, where parquet is smaller by roughly an order of
magnitude and round-trips dtypes exactly. **ECSV is the better default for
anything meant to be read or hand-edited** -- astropy's plain text with a YAML
header carrying the column types, so it stays inspectable without giving up what
CSV gives up. Plain CSV and TSV are accepted so an existing catalog needs no
conversion step.

The pipeline only cares about four columns -- `picid`, `catalog_ra`,
`catalog_dec`, `catalog_vmag` -- so whatever produces the catalog is free to
change without touching this package.

The one thing plain CSV cannot carry is dtypes, and `picid` is an identifier: a
single blank turns the column into floats that then join as `1234.0` against
integer ids and match nothing. `read_catalog` checks it and fails there rather
than later. It then returns `picid` as a categorical, because it names a star
rather than measuring one, and a filtered result prunes its unused categories --
otherwise a field cut from an all-sky catalog carries every id in the sky.

The cost was accepted knowingly: this discards working archive integration that
would have to be rewritten if bulk cloud processing is ever wanted again. What it
buys is that the offline path is the *only* path, so it cannot rot quietly while
the cloud path is the one being exercised -- and that running the algorithm needs
no credentials, which is what 4 requires of a citizen-science project.

`panoptes-data` remains how sequences are discovered and fetched (4.6). That is a
step that puts files on disk before the pipeline runs, not a runtime dependency
of it.

### 4.4 Notebooks, if they come back

`notebooks/` is gitignored in full, and the two files that were tracked by
exception -- `ProcessFITS.ipynb` and `ProcessObservation.ipynb` -- have been
deleted. Nothing in the repository is a notebook.

If notebooks return they import the library and plot; they do not hold the
algorithm. The reason is in conformance audit 5.0 and 5.13: logic consequential
enough to invalidate every measurement in this document sat in a cell where no
test could reach it.

Note for anyone reading an older version of this section: the working copies
described there (`notebooks/working/`, the `RunProcessFits-Copy*` variants, the
`Untitled` files) were never in the repository. They are one person's local
directory, covered by the `notebooks/` ignore rule, and no repository cleanup
reaches them.

### 4.5 Test data and the archive

Three tiers, and only the first belongs in git.

**Tier 1 -- committed fixtures, under ~5 MB, plain git, no LFS.** A trimmed
slice of a real observation: 40 sources x 40 frames x 180 pixels in float32 is
about 1.2 MB. Enough for unit tests and CI, small enough that nobody thinks
about it. **Do not cut one until 3.1 lands** -- a fixture built from today's
stamps would bake the un-subtracted background into the test suite permanently.

**Tier 2 -- working datasets, GB scale, local only, never committed.** What gets
committed is a manifest: sequence IDs, frame counts, checksums, and the
`panoptes-data` call that fetches them. That buys reproducibility without the
bytes, and a manifest diff is readable in a way a binary diff is not.

**Tier 3 -- the full archive on the processing server**, over VPN, for parameter
sweeps and final runs.

**Skip git-lfs.** It exists for data with no other home. This data has a home,
an owner and a client library, so LFS would duplicate the archive at the cost of
setup friction, bandwidth quota, and a repository nobody can clone cheaply.
Revisit only if some dataset becomes load-bearing for CI *and* cannot be trimmed
to tier 1 -- and prefer trimming.

**Develop locally, sweep on the server.** Running only on the server is the
tempting shortcut and the wrong one: CI cannot reach it, and the code quietly
becomes unable to run anywhere else. Keeping tier 1 and tier 2 working locally
keeps the offline-first goal in 4 honest, and the server stays what it should
be -- where the expensive runs happen.

### 4.6 The data layer and camera profiles

Superseded by the [data contract](data-contract.md), which this section's cheap
test resolved.

The test was: write one real selection query for dataset B against
`panoptes-data` and see whether it fits. It does not, and the reason is more
useful than the verdict. The client's model is fine; the *published index* is a
per-observation flattening that drops the aggregates selection needs, and the
per-frame records behind it are reachable only one sequence at a time. Duration
is not expressible at all, and `num_images` counts frames uploaded rather than
usable. See data contract 9.

The direction of flow is what this section had backwards. `panoptes-data`
queries what *this* pipeline produced, so it is downstream and follows rather
than constrains. Extend it; the question is settled.

Camera profiles move with it. The registry only ever has to carry what no
header supplies -- gain, read noise and black level -- and it belongs here
rather than in the read client. Gain in particular is a POCS constant this
repository copied, so the fix is upstream. See data contract 6.

### 4.7 CI

`.github/` is empty. Add: ruff, pytest, and a smoke run of
`scripts/benchmark_lightcurve.py` against a trimmed fixture, so a regression in
the numbers fails the build rather than being discovered months later.

## 5. Sequencing

**First, and blocking** -- 3.1. Subtract the background, then reprocess one
observation. Until that lands, every measurement in this plan is diluted by an
unknown factor and no result can be defended.

**Then, and probably the big one** -- 3.2. Start with its diagnostic (how much
variance the measured drift explains) and the rotation-versus-translation split,
because those rank the rest of 3.2 on evidence rather than argument.

**Alongside** -- 3.3 on the production path, 3.4 alpha sweep, 3.6 superpixel
apertures. All implemented or nearly so, and all measurable against benchmark 1
the moment 3.1 is done.

**Next** -- 3.5 per-channel fitting, 4.1-4.3 library and CLI.

**After** -- 3.7 flat fields, benchmark 4.

Deliberately deferred: anything that improves throughput before precision is
established, and any move to fit transit parameters. Get one target right
first.

## 6. Risks

**Precision is much further away than the first measurement suggested.** The
honest single-channel number is 4.8%, not the sub-percent figures the
un-subtracted stamps produced. Some of that gap is the stopgap sky subtraction
and will close with 3.1, but the distance to 0.5% should be assumed large until
measured on properly reduced data.

**The 0.5% goal may not be reachable from a single camera.** The paper states
its result "approaches the fundamental noise floor possible from a single
camera". If the floor ratio in 1.2 comes back near 1, the remaining paths are
combining units or longer exposures, not algorithm work. Measuring the floor
ratio early (issue #141) tells us which problem we are solving, and should happen
before any large effort is spent on 3.5.

**Overfitting is the standing hazard.** 100 free coefficients fit to one
target's pixel history will happily absorb a transit. Suppression (1.2) is
reported on every run for this reason, and no precision result should be
accepted without it.

**Fixing correctness can make the numbers look worse.** Correcting the target
self-reference defect (conformance audit 5.3) removes a bias *toward* flat
lightcurves, so measured RMS may rise. That is a real improvement presenting as
a regression, and it needs to be noted when it happens rather than debugged.
