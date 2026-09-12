# Algorithm design

What the algorithm *is*, stated independently of any implementation, and the
architecture that follows from it. Cite as "algorithm design 2.1".

The published paper is a building block, not a specification. Where its choices
and the idea diverge, the idea wins.

---

## 1. The idea

A wide field gives any target many candidate reference stars whose light lands
on the colour filter array the same way the target's does. Those references can
be combined into an idealised star that should behave exactly as the target
would absent a transit. The difference is then zero unless the target's flux has
genuinely changed.

### 1.1 As a model

For star *k*, frame *i*, pixel *j*:

```
P[k,i,j] = F[k,i] * S[k,i,j] + B[i,j] + noise
```

`F` is the true flux -- the thing we want. `S` is the normalised spatial
profile: how that flux lands on pixels, summing to 1 over *j*. `B` is
background.

`S` depends on per-star properties `θ` (sub-pixel position on the Bayer grid,
colour, field position and hence aberration) and per-frame state `φ` (pointing,
focus, seeing). The systematic is that `S` moves with `φ` in a way that depends
on `θ`, so a fixed aperture sum is not proportional to `F`.

**The load-bearing insight**, and the paper's real contribution: dividing a
stamp by its own summed flux marginalises `F` out and leaves an estimate of `S`
alone. Shape and brightness separate. Match on shape, difference the
brightness.

Everything else is implementation, and open.

### 1.2 What this implies about transits

A transit is a change in `F` with no change in `S`. So a profile model can be
fit across every frame, in-transit ones included, without absorbing the signal.
That is what makes the approach safe, and it holds for any implementation that
keeps the normalisation. It must still be verified by injection every time, not
assumed.

## 2. Where the paper's implementation and the idea diverge

Three of its choices are not required by the idea, and measurement says all
three cost performance.

### 2.1 Near-neighbour matching is the wrong tool

The paper searches for the most similar references and combines the top 100. It
then observes that near-neighbours are rare -- only 3% of references fall within
half the similarity radius -- calls this the curse of dimensionality, and
compensates with negative coefficients, i.e. extrapolation.

That is a symptom of the method, not of the data. Measured on
`PAN007_f6eb3d_20250930T030402` across 900 usable stars: the profile manifold
needs a **median of 4 components for 95% of the variance**, in a 180-pixel
space. The paper's own pseudo-dimension estimate was ~5.

If the manifold is that low-dimensional, you do not need neighbours. You need
*coverage*, and 3,000 stars cover a 5-dimensional manifold easily while
near-neighbours in 180 dimensions stay rare. Learn the manifold once from every
star, place the target on it, predict its profile. That replaces 100 free
parameters per target with roughly 5 latent coordinates plus a shared model.

Caveat, and it is not small: the rank is frame-dependent. Some frames need 1-4
components, others 18-36. That instability has to be explained -- genuinely
different PSF state, bad frames, or an artefact of the stopgap sky subtraction
-- before the design is built on it.

### 2.2 The similarity metric partly measures brightness

The variance of a profile estimate scales as 1/N, so a brighter reference scores
better simply by being less noisy. The squared-difference score carries that
offset and the ranking inherits it.

Measured: median similarity score falls monotonically across reference
brightness quintiles, from 4.48 for the faintest to 2.59 for the brightest, and
**71% of selected references are brighter than their target** where a
brightness-neutral metric would give 50%.

So "most morphologically similar" is substantially "brightest available". A
noise-weighted distance fixes it directly. In a manifold model the problem
largely dissolves, because a model is fit rather than neighbours ranked.

### 2.3 One step is doing two unrelated jobs

Dividing target by comparison removes the pixelisation systematic *and*
atmospheric transparency at once. These are different: pixelisation is a change
in `S`, transparency is a genuine common change in `F`. Conflating them forces
one coefficient vector to be simultaneously shape-matched and flux-
representative, and nothing guarantees a shape-optimal vector is a low-variance
flux estimator -- large opposing coefficients can cancel in shape while
amplifying flux noise.

Separate them. A profile model handles `S`. A plain ensemble handles
transparency.

## 3. Proposed architecture

1. **Background subtraction at frame level.** Not stamp level -- measured, the
   sky is flat across a stamp and there is nothing to fit there.
2. **Superpixel-aligned stamps, re-centred per frame** on the nearest superpixel.
   Whole-superpixel shifts preserve Bayer phase, so bulk drift is absorbed and
   only sub-pixel phase remains.
3. **Normalise** each stamp by its own sum to estimate `S`.
4. **Learn a low-rank profile model** across all stars and frames, with the
   target excluded or down-weighted.
5. **Predict the target's profile** for each frame from its latent coordinates.
6. **Optimal extraction** (Horne 1986) of the target's flux using that predicted
   profile and the noise model -- a minimum-variance estimate, robust to drift
   and aperture loss by construction, rather than a hard-edged aperture sum.
7. **Divide by an ensemble** of comparison stars extracted the same way, to
   remove transparency.

What this buys, beyond precision: about 5 free parameters per target instead of
100, so far less overfitting and far less scope for signal absorption; faint
stars become tractable, where a per-star 100-parameter fit never was; drift
becomes a frame-level latent the model absorbs rather than something matching
must accidentally cancel; and per-point uncertainties fall out of optimal
extraction for free.

**This is a proposal to test head-to-head, not a conclusion.** Same data, same
metrics, against the paper's method. The infrastructure to do that already
exists.

## 4. Two objectives, at two layers

These are not the same thing, and conflating them is how an algorithm gets
quietly tuned toward its own prior.

### 4.1 The algorithm's objective: fidelity

**The algorithm's job is to return the target's true relative flux -- nothing
suppressed, nothing invented.** That is it. It should know nothing about
transits.

Optimising the algorithm for transit detection would bias it toward signals
shaped like the assumed transit and against everything else in the data:
long-duration events, stellar variability, flares, anything the survey has not
thought of. It is also circular, since the same machinery later claims the
detections.

Measure it as a **signal transfer function**: inject sinusoids across a range of
periods and measure recovered amplitude over injected amplitude.
`injection.transfer_function` does this. A transfer of 1.0 at every timescale of
interest is what fidelity means quantitatively, and sinusoids are used rather
than transits precisely because they carry no assumption about signal shape.

The failure mode to watch is a **long-timescale rolloff**. Any model with many
free parameters fitted across a whole sequence will absorb slow variation, and a
transit-shaped test at one duration can miss that entirely.

Fidelity is one axis; residual noise is the other, measured with the scatter and
red-noise metrics in `metrics`. A change is an improvement only if it lowers
noise **without** lowering transfer. Reporting one without the other is how
signal suppression gets sold as precision.

### 4.2 The project's objective: detection

The survey is judged on **injection-recovery completeness at a fixed false-alarm
rate**, over a grid of depth, duration and phase. `detection.completeness` does
this, with the threshold from `detection.false_alarm_threshold`.

That metric belongs to the end-to-end survey, not to the algorithm, and the
noise that matters for it is on transit timescales -- roughly 1 to 4 hours --
rather than at an arbitrary 30 minutes.

Scatter alone is a proxy for neither objective, and a poor one wherever the
noise is correlated.

## 5. Status of the measurements in this document

Every measurement quoted above came from
`notebooks/PAN007_f6eb3d_20250930T030402/observation.h5`, a product of the
existing pipeline. That pipeline is known to carry an un-subtracted sky
pedestal (conformance audit 5.0), cuts stamps at a fixed mean position with an
arbitrary size, and may carry further systematics introduced in processing. **It
is not a trustworthy substrate, and the rebuild starts from raw frames.**

Which findings survive that, and which do not:

| Finding | Basis | Status |
|---|---|---|
| Background computed then discarded | reading `ProcessFITS.ipynb` cell 18, confirmed by pixel values | **solid** -- a property of the code |
| Similarity metric rewards brightness | analytic (profile variance ~ 1/N), confirmed by measurement | **solid** -- the derivation does not depend on the data |
| Target used as its own reference | reading the notebook, confirmed by reproducing `to_xarray` ordering | **solid** |
| RGB masks transposed for non-square stamps | reading the code | **solid** |
| Profile manifold needs ~4 components | measured on these stamps | **provisional** -- inherits the pedestal, the stamp cut, and an unexplained frame-to-frame rank instability |
| Coefficient fit gains 1.15x | measured on these stamps | **provisional** -- rests on a stopgap sky subtraction that over-subtracts stellar wings |
| Greens on the diagonal (GRBG/GBRG) | measured on these stamps | **describes this file only** -- phase depends on how stamps were cut, so a new pipeline must re-derive it with `masks.infer_pattern` |

The two load-bearing claims for the architecture in 3 -- that the manifold is
low-dimensional, and that a profile model beats neighbour matching -- are both
in the provisional row. Re-measure them on stamps cut from raw frames before
committing to the design. If the manifold is not low-rank on clean data, section
3 does not follow.

## 6. Partial transits and the network

The survey targets long-period planets, so a transit can last longer than any
one observation. The end goal is to combine segments from units at different
longitudes -- Los Angeles catching the first few hours, Hawaii the middle with
overlap, Japan the egress -- into a single event.

That is a high-level goal, but it constrains the algorithm now, because several
conventions that are harmless for short transits are fatal for partial ones.

### 6.1 No self-normalisation

Paper Eq. 6 normalises the lightcurve so its median is unity. For a transit
comfortably inside a long baseline that is harmless. For a window that is
wholly or partly in transit it subtracts the signal from itself: a fully
in-transit segment normalises to a flat line and the depth is gone.

`differential_lightcurve` therefore defaults to `normalize=False`. The ratio of
target to comparison ensemble is already a relative flux; the ensemble sets the
scale, and nothing further is needed. Depths are measured as a fraction of a
fitted baseline, which is invariant under rescaling, so two units on different
scales still agree on depth.

### 6.2 The output must be transferable

A segment is only stitchable if it measures a quantity another unit also
measures. That means target flux relative to a **defined, shared** comparison
ensemble, not to the target's own history. Two units observing the same field
can use the same catalogue stars, so their ratios differ only by a constant.

Each segment therefore has to carry enough metadata to be tied to another:
which comparison stars, which camera and bandpass, airmass, and the time system.

### 6.3 Offsets are solved, not assumed

Different bodies have different effective bandpasses, so the target-to-ensemble
ratio carries a constant factor set by the colour difference between target and
ensemble. That factor is constant per unit for a given target and ensemble, so
it can be calibrated -- but only against something.

**Overlap is what makes the offsets identifiable.** With overlapping segments,
the per-segment offsets are constrained by the data. Without overlap they are
degenerate with the signal itself, and a depth can be traded against an offset
with no way to tell which is which. This is the familiar structure of a global
fit with per-segment nuisance parameters, and the practical consequence is that
scheduling must guarantee overlap rather than merely aim for continuity.

### 6.4 Fidelity beyond the observation length is the point

The long-timescale rolloff flagged in 4.1 is not an edge case here -- it is the
primary scientific requirement. Any model with many free parameters fitted
across a sequence will absorb variation on the timescale of that sequence, which
is precisely the timescale a partial transit lives on.

So `injection.transfer_function` must be probed at periods reaching **beyond**
the observation duration, and the rolloff there quantified rather than assumed
away. An algorithm with excellent short-timescale fidelity and a rolloff at the
night length would look good on every conventional metric and be useless for
this survey.

### 6.5 Times in BJD_TDB

Combining across sites and epochs needs barycentric dynamical time, not UTC.
The inter-site light-travel difference is negligible (~40 ms across Earth), but
the barycentric correction over months is not, and ephemerides depend on it.
Adopting it now is cheap; retrofitting a time system through a processed archive
is not.

### 6.6 Differential extinction, and a synergy

Each site observes the target at a different airmass, and second-order
extinction scales with the colour difference between target and comparison
stars. In a self-normalised lightcurve this hides inside the normalisation. In
a transferable one it has to be handled.

The method helps here, by accident. Selecting references on their Bayer
morphology implicitly selects on colour, since colour is what drives how a star
samples the filter array. Morphologically matched comparison stars are therefore
also colour-matched, which is exactly the condition that minimises second-order
extinction. An under-appreciated benefit of the approach for this goal.
