# Conformance audit

How closely the code in this repository implements the algorithm published in
Gee et al., *On-sky Demonstration of Precision Photometry with Bayer Color
Filter Arrays*.

Audited at commit `dd813b2` (branch `pipeline-working`). Cite findings as
"conformance audit 5.3".

---

## 1. Summary

Two findings, in order of impact.

**The background is computed and then thrown away.** `ProcessFITS.ipynb` cell
16 computes `reduced_data = data - bg_data` exactly as paper section 4.2
describes, and cell 18 then writes `dict(raw=raw_data)` to
`reduced_filename`. The background-subtracted array only reaches the *extras*
file, and only when `save_extras=True`, which defaults to `False`. Every
postage stamp in every `observation.h5` therefore carries bias plus sky.
Measured on `PAN007_f6eb3d_20250930T030402`: the minimum pixel value across all
stamps is 571 ADU and the median is 770, against a bias of 512. For a median
source, the star is under 3% of its own stamp sum. This is a regression from
the published pipeline, which does subtract a three-channel global background
before cutting stamps. See 5.0.

**The published algorithm's central step is not implemented.** Its first two
steps are, and they are correct. Building a synthetic comparison star from a
fitted linear combination of references -- paper sections 3.2.3 and 3.2.4 -- is
absent. The code substitutes an unweighted arithmetic mean of the top 100
reference stars, which is ordinary ensemble differential photometry; the
linear combination in normalized space is what lets the comparison reproduce
the target's *specific* interaction with the Bayer pattern.

A least-squares solver matching paper section 3.2.3 did exist in this
repository and was deleted in the PyScaffold reorganization (it survives in
history at `src/panoptes/pipeline/utils/processing.py`, functions
`get_ideal_full_coeffs` and `get_ideal_full_psc`). Even that version used
unregularized `scipy.linalg.lstsq`; the regularization implied by paper
Figure 7 has never existed in code.

The two findings interact, and the order matters. Measured on stamps *as
stored*, restoring the coefficient fit appears to improve red-channel scatter
by 2x. Remove the sky pedestal first and that collapses to 1.15x -- because on
un-subtracted stamps the fit is largely matching smooth background structure
between neighboring stars, not stellar morphology. Numbers in improvement
plan 2.3.

## 2. Where the algorithm lives

| Paper section | Implemented in | Kind of file |
|---|---|---|
| 4.2 Background subtraction | `notebooks/ProcessFITS.ipynb` cells 13-16 | notebook |
| 4.3 PSC creation | `notebooks/ProcessObservation.ipynb` cells 49-57 | notebook |
| 3.2.1 Prepare PSCs | `notebooks/working/MakeLightcurves.ipynb` cells 14-15 | notebook |
| 3.2.2 Find reference stars | `notebooks/working/MakeLightcurves.ipynb` cells 23-25 | notebook |
| 3.2.3 Determine coefficients | *nowhere* | -- |
| 3.2.4 Build comparison star | *nowhere* | -- |
| 3.2.5 Differential photometry | `notebooks/working/MakeLightcurves.ipynb` cell 45 | notebook |

Every algorithmic step lives in a Jupyter notebook. Nothing under `src/` does
any photometry: those modules handle orchestration, cloud I/O, plate solving,
catalog matching and source detection only. The lightcurve notebook is not even
on the path the CLI expects (see 5.14).

This is the structural finding behind improvement plan 4. Notebooks cannot be
unit-tested, cannot be diffed meaningfully in review, and silently carry stale
execution state. Two of the defects below (5.3, 5.6) are the kind that only
survive in code nobody can test.

## 3. Core algorithm conformance

### 3.1 Prepare PSCs -- conforms

Paper Eq. 1 normalizes each frame by its own summed flux.
`MakeLightcurves.ipynb` cell 15 does exactly this. Correct.

### 3.2 Find reference stars -- conforms

Paper Eq. 2 sums the squared difference between normalized target and
reference over pixels, then over frames. Cell 23 does this. Correct.

The notebook additionally computes per-channel scores and takes the
intersection of the top-100 sets across R, G and B (cell 27) -- a reasonable
extension, but the result is then discarded (see 5.6).

### 3.3 Determine coefficients -- **not implemented**

Paper Eq. 4 minimizes the residual between the normalized target and a linear
combination of the normalized references, solving for one coefficient per
reference. There is no solver anywhere in the current tree. `scipy` is imported
by exactly one file, `notebooks/working/MakeLightcurves-Copy1.ipynb`, and is
unused there.

The paper says "in practice a regularization term can also [be] applied", and
Figure 7 shows 46 of 100 coefficients at exactly zero. Exact zeros are the
signature of an L1 penalty; plain least squares does not produce them. So the
published configuration is a *regularized* fit whose regularizer and strength
are not stated in the paper. This is an open question, not just a porting job
-- see improvement plan 3.4.

### 3.4 Build comparison star -- **not implemented**

Paper Eq. 5 applies the normalized-space coefficients to the flux-carrying
reference stamps. `MakeLightcurves.ipynb` cell 45 instead computes:

```python
diff_lc = target_psc.sum(1) / ref_cube[1:use_num_refs].mean(0).sum(1)
```

An unweighted mean. Equivalent to forcing every coefficient to `1/r`.

### 3.5 Differential photometry -- partially conforms

Paper Eq. 6 sums target and comparison flux *within an aperture* and ratios
them, per color channel, giving three lightcurves (paper Figure 12).

Cell 45 sums the **whole stamp** and produces **one colorless lightcurve**. The
RGB masks built in cell 10 and the per-channel normalized cubes built in cell
41 are never used in the photometry. Median normalization to unity is applied
correctly.

## 4. Pre-processing conformance

### 4.1 Background subtraction -- computed correctly, then discarded

The computation conforms: per-channel `photutils.Background2D` with no
interpolation across the Bayer pattern, as described in paper section 4.2, and
a box size matching the paper's 79x84. Two problems follow.

**The result is not saved** (5.0). `reduced_data` is computed and dropped.

**The low-resolution median filter is 3x3 in `settings.py`; the paper used
11x12.** A 3x3 filter over a 44x62 box grid smooths far less, so more
small-scale background structure survives -- which matters more, not less,
once the subtraction is actually applied.

### 4.2 PSC creation -- conforms

`bayer.get_stamp_slice` is used to force superpixel-aligned stamps, and mean
catalog positions across all frames set the center superpixel, as in paper
section 4.3. Spot-checked on `PAN007_f6eb3d_20250930T030402`: all stamp origins
are even. Correct.

### 4.3 Stamp geometry -- diverges, and is unsafe

The paper uses a fixed 10x10 stamp (n = 100). `ProcessObservation.ipynb` cell
49 chooses 10 or 18 per axis independently, based on measured drift, so stamps
can be **non-square**. The observation checked here is 10x18 = 180 pixels.

Non-square stamps are legitimate, but they break the downstream mask code
(see 5.4). There is also no guard that a stamp has an even size in both axes,
which is required for the Bayer phase to be well defined.

### 4.4 Catalog limits -- diverges

Paper section 4.3 builds PSCs for 6 < mV < 10. `settings.py` uses 6-13,
`ProcessObservation.ipynb` uses 6-12. A wider pool is probably good (more
references) but it is an untested change from the published configuration and
directly affects reference selection.

### 4.5 Camera constants -- diverge, and are global

Paper Table 2, Canon EOS 100D: bias 2048 ADU, saturation 11535 ADU (bias
removed), gain 1.50 e-/ADU. `settings.py` hardcodes `zero_bias=512`,
`saturation=15872`, `effective_gain=1.5` for **all** cameras. PANOPTES is a
heterogeneous fleet by design; these belong in per-unit configuration. Wrong
bias propagates directly into Eq. 1, because normalization of a
bias-contaminated stamp is not the same as normalization of a clean one.

## 5. Defects

Ordered by impact on photometric precision.

**5.0 -- Background-subtracted data never reaches the stamps.**
`ProcessFITS.ipynb` cell 18 writes `dict(raw=raw_data)` to `reduced_filename`;
`reduced_data` goes only to `extras_filename`, gated on `save_extras`, which
defaults to `False`. `ProcessObservation.ipynb` cell 57 then cuts stamps from
that file. Consequences:

- Paper Eq. 1 normalizes a stamp that is mostly flat pedestal, so the
  "morphology" being matched is dominated by background, not by the star.
  Reference selection and the coefficient fit are solving the wrong problem.
- Differential photometry ratios `(star + sky) / (star + sky)`, so every
  measured transit depth is diluted. Measured dilution on
  `PAN007_f6eb3d_20250930T030402`: 2.9x for a bright target, 44x for a median
  one.
- Fractional RMS computed on these stamps is not photometric precision. Faint
  sources appear to have *lower* scatter than bright ones, because their stamp
  sum is a stable sky pedestal.

Independent confirmation of the diagnosis: the temporal per-pixel scatter of
sky pixels is 10.4 ADU (robust). For a 780 ADU pedestal made of 512 ADU bias
plus ~268 ADU of sky at a gain of 1.5 e-/ADU, photon noise on the sky alone
predicts 13 ADU. Bias contributes no photon noise, so the measured scatter is
only consistent with the pedestal being bias plus sky -- that is, with raw data.

`lightcurve.subtract_stamp_sky` is a stopgap for already-processed data; the
fix is to write `reduced_data` to `reduced_filename`.

**5.1 -- The comparison star is an unweighted mean, not a fitted combination.**
Paper sections 3.2.3 and 3.2.4 are absent. See 3.3 and 3.4 above. This is the
single largest gap.

**5.2 -- Regularization has never been implemented.** The only solver that ever
existed in this repository used plain `scipy.linalg.lstsq`. Paper Figure 7
cannot be reproduced by that code.

**5.3 -- The target is used as one of its own references.** `ref_cube` is built
via `to_xarray()` (cell 39), which sorts axis 0 by PICID. Cell 45 then slices
`ref_cube[1:]`, assuming axis 0 is similarity rank with the target at index 0.
Verified: `to_xarray()` returns PICID-sorted order, so `[1:]` drops whichever
reference has the lowest PICID and retains the target. A target that partly
compares against itself is biased toward a flat lightcurve -- it suppresses
real transits.

**5.4 -- RGB masks are transposed for non-square stamps.** `make_stamps`
(`utils/observations.py`) slices `data[y_min:y_max, x_min:x_max]` and ravels
row-major, so the flat pixel axis unpacks as `(height, width)`.
`MakeLightcurves.ipynb` cell 10 calls `make_masks(stamp_size=(stamp_width,
stamp_height))` -- the other way round. For the 10x18 stamps in
`PAN007_f6eb3d_20250930T030402` the element count still matches, so it
broadcasts silently and assigns colors to the wrong pixels. Live on current
data.

**5.5 -- Two Bayer mask conventions, never cross-checked.**
`ProcessFITS.ipynb` uses `panoptes.utils.images.bayer.get_rgb_masks`;
`MakeLightcurves.ipynb` hand-rolls its own with hardcoded slice offsets. The
hand-rolled version also follows the `numpy.ma` polarity (`True` means
*excluded*) while reading as a selection mask. Nothing verifies the two agree.

Confirmed empirically, and it bites. The two greens of a Bayer quad share a
filter, so their sky levels match; measured on the star-free stamps of
`PAN007_f6eb3d_20250930T030402` the matching pair sits on the **diagonal**
(768 and 765 ADU) while the other two differ (699 and 642). The pattern in
stored data is therefore GRBG or GBRG, and any code assuming RGGB assigns one
green to "red" and mixes the true red and blue into "green". Work in this
session made exactly that mistake before catching it, which is the argument for
`masks.infer_pattern` deriving the phase from data rather than any module
declaring a convention. Red versus blue is not recoverable from sky levels and
needs the `MEASRGGB` header, which `extract_metadata` already parses.

**5.6 -- The per-channel reference selection is computed and then thrown away.**
Cell 27 intersects the R, G and B top-100 sets; cell 28 sets `num_refs` to the
size of that intersection; cell 40 then indexes `top_matches.index[:num_refs]`
-- the *monochrome* ranking. The result is the mono top-N truncated to an
arbitrary length.

**5.7 -- No aperture, and no per-color lightcurves.** See 3.5. The paper's
headline result is three RGB lightcurves; the code produces one full-stamp
curve.

**5.8 -- A naive aperture would break the color ratio anyway.** Measured on a
10x10 stamp with a radius-2 circular aperture: as the centroid shifts by half a
pixel, the green:red pixel ratio inside the aperture swings from 0.5 to 4.5.
Since the star drifts by ~1 pixel per period (paper Figure 5), a hard-edged
aperture introduces a color-dependent flux modulation at exactly the tracking
period. Any aperture must be grown to whole superpixels or weighted. Encoded as
a regression test in `tests/test_lightcurve_masks.py`.

**5.9 -- Background median filter is 3x3, not the paper's 11x12.** See 4.1.

**5.10 -- Duplicate catalog matches are resolved arbitrarily.**
`images.match_sources` drops duplicates with `keep='first'` and a comment
claiming results "are returned in order of catalog separation". They are not:
`get_catalog_match` never sorts by `catalog_sep`. The retained detection is
whichever came first in detection order.

**5.11 -- `detect_sources` raises whenever any pixel is saturated.**
`if not reduced_data.mask:` evaluates an ndarray in boolean context. Verified:
raises `ValueError: The truth value of an array with more than one element is
ambiguous` as soon as one pixel is masked -- which is the normal case for any
field with a bright star. The following line, `np.zeros_like(reduced_data.mask,
dtype=bool)`, is also wrong when the mask is `numpy.ma.nomask`: it produces a
scalar.

**5.12 -- Background subtraction is implemented twice, differently.**
`utils/images.subtract_background` masks with `bg.mask`;
`ProcessFITS.ipynb` cell 15 masks with `bayer.get_rgb_masks(data)`. The library
function is dead code that no longer matches the notebook it was extracted
from.

**5.13 -- `ProcessObservation.ipynb` does not run.** Cell 25, which builds
`sources_df`, is commented out. Cell 26 then references the undefined name.

**5.14 -- The CLI points at notebooks that do not exist at those paths.**
`cli/main.py` references `notebooks/ProcessFits.ipynb` (actual:
`ProcessFITS.ipynb`) and `notebooks/MakeLightcurves.ipynb` (actual:
`notebooks/working/MakeLightcurves.ipynb`). The first is a case-only mismatch:
it works on macOS and fails in the Linux container.

**5.15 -- Camera constants are global, not per-unit.** See 4.5.

**5.16 -- Similarity search is O(p^2).** Every target is scored against every
other source. Flagged in paper section 3.2.2 as future work; with ~3,200
sources per frame this dominates runtime for whole-observation processing.

### Fixed on branch `algorithm-v2`

These were repo-health blockers rather than algorithm defects, and were
corrected while setting up the test harness:

- `pytest` could not start at all: `--test-databases all` in the `addopts` of
  `pyproject.toml` is not a recognized option.
- `pip install -e .[dev]` failed: `jupyterlab_beta` does not exist on PyPI.
- `ruff` was configured for `py38` while `requires-python` is `>=3.12`.

## 6. Repository state

- **Tests**: none. `tests/conftest.py` contains only a docstring.
- **CI**: `.github/` exists but is empty.
- **Execution**: the algorithm cannot run without Firestore, BigQuery and GCS.
  There is no path to process a local directory of FITS files end to end.
- **Notebooks**: 20 files under `notebooks/working/`, including six
  `RunProcessFits-CopyN.ipynb` variants and two `Untitled` notebooks. Which one
  produced any given published figure is not recoverable.
- **Branches**: `develop`, `prepare-cleanup` and `pipeline-working` have
  diverged; `pipeline-working` carries uncommitted changes and untracked
  deployment files.
