# panoptes-pipeline

Differential photometry for PANOPTES: recovering exoplanet transits from DSLR
wide-field images with a Bayer color filter array.

## The idea

A wide field gives any target many candidate reference stars whose light lands
on the color filter array the same way the target's does. Those references
build an idealized star that behaves as the target would absent a transit, so
the difference is zero unless the target's flux genuinely changed.

The load-bearing insight: dividing a stamp by its own summed flux marginalizes
brightness out and leaves the spatial profile. Shape and brightness separate.
Match on shape, difference the brightness.

The idea and the network are the same principle at two scales, and low unit
cost enables both: many stars per unit calibrate the systematic, many units
cover a transit longer than one night. Statistical power from multiplicity,
where multiplicity is cheap. Heterogeneity is a consequence of that, not a
nuisance, and any approach needing a characterized or uniform fleet contradicts
the premise. See algorithm design 1.3.

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

**1. Near-neighbor matching is probably the wrong tool.** The profile manifold
measures as roughly 4 components for 95% of the variance in a 180-pixel space.
If that holds on clean data, you want coverage of a low-dimensional manifold,
not the 100 nearest neighbors -- which is why the paper hit a curse of
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

## Transits can be longer than an observation

The survey targets long-period planets and the end goal is stitching segments
from units at different longitudes into one event. So **never normalize a
lightcurve to unit median** -- for a window that is wholly or partly in transit
that subtracts the signal from itself. `differential_lightcurve` defaults to
`normalize=False`; the comparison ensemble sets the scale. Depths are fractions
of a fitted baseline, which survives rescaling.

Consequences: output must be relative to a defined, shared comparison ensemble,
not to the target's own history; per-unit offsets are solved against overlapping
segments, and without overlap they are degenerate with the signal; fidelity at
periods *beyond* the observation length is the requirement, not an edge case;
times want BJD_TDB. See algorithm design 6.

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
- `src/panoptes/pipeline/` (the rest) -- `utils/images.py` (calibration, source
  detection, plate solving, catalog matching; reused rather than rebuilt, see
  improvement plan 4.1), `utils/sources.py` (catalog matching against a local
  parquet), plus `settings.py`, `utils/observations.py` and `utils/plot.py`.
- `plans/`, `scripts/`, `tests/`.

**There is no cloud code path and no notebooks.** Firestore, BigQuery, GCS, the
papermill execution path, the FastAPI service, the console script and the Docker
image were all deleted; `notebooks/` is gitignored in full. The catalog is a
local file named by `params.catalog.catalog_filename` -- parquet, ECSV or CSV,
by suffix -- and `get_stars` fails loudly when it is unset rather than reaching
for the network. The pipeline requires exactly four catalog columns (`picid`,
`catalog_ra`, `catalog_dec`, `catalog_vmag`), so the catalog can be rebuilt
without touching this package. `picid` is a categorical: it names a star rather
than measuring one, so group on it with `observed=True`. A CLI comes back with improvement plan 4.2. See
improvement plan 4.3 for why removal beat adapters.

## Running things

Everything goes through `uv`. `uv sync` installs the project plus the `dev`
dependency group, so the package is importable and no `PYTHONPATH` is needed:

```bash
uv sync                    # project + dev tooling
uv run pytest              # 139 tests
uv run ruff check .        # lint
uv run ruff format .       # format
```

There are no extras. Every module under `src/panoptes/pipeline/` imports from a
plain `uv sync` -- there is no environment in which part of the package works and
part raises `ModuleNotFoundError`.

```bash
uv build                   # sdist + wheel into dist/
uv publish                 # needs UV_PUBLISH_TOKEN
rm -rf build dist          # clean
```

Standalone scripts are PEP 723 and declare their own dependencies, so they run
against a throwaway environment rather than the synced one:

```bash
uv run scripts/benchmark_lightcurve.py OBS.h5 --channel r --sky-subtract --held-out
uv run scripts/survey_targets.py OBS.h5 --sky-subtract
```

## Measuring a change

**Two objectives at two layers, and they are not the same.** See algorithm
design 4.

The *algorithm's* job is to return the target's true relative flux -- nothing
suppressed, nothing invented. It should know nothing about transits; optimizing
it for a transit shape biases it toward that prior and against everything else
the data holds. Score it with `injection.transfer_function`: sinusoids injected
across a range of periods, recovered amplitude over injected. 1.0 at every
timescale is the goal, and a long-timescale rolloff is the failure mode -- any
model with many free parameters fitted across a whole sequence will eat slow
variation. Report transfer alongside noise, always: a change that lowers scatter
while lowering transfer has suppressed signal, not removed noise.

The *project's* objective is detection, scored with
`detection.completeness` against a `detection.false_alarm_threshold` (built from
circular time shifts, which preserve red noise). That belongs to the end-to-end
survey, not to the algorithm.

Also required of any precision claim:

- Inject at the **pixel level**, before normalization, so the whole pipeline
  sees the signal. Injecting into a finished lightcurve measures nothing.
- Score on **held-out frames**. Fitting many free parameters per target will
  absorb noise given the chance.
- Report the **per-target win rate** across a sample, not one target's median.
  A single flattering target is how the first baseline here came out 10x too
  optimistic.

Correcting a defect can make numbers look worse -- removing a bias toward flat
lightcurves is a real improvement presenting as a regression. Expect it.

## Versioning and tags

[Semantic versioning](https://semver.org). `hatch-vcs` derives the package
version from the nearest `v*` tag, so **the tag is the only source of truth** --
never hardcode a version anywhere.

**We are pre-1.0, deliberately.** Under SemVer, `0.y.z` means the public API is
not stable and may break in any release. That is an accurate description of a
package whose cloud modules are being deleted, whose CLI is being rewritten, and
whose core algorithm is pending a head-to-head test. Do not reach for 1.0.0 to
signal that the project is serious; reach for it when the API stops moving.

While pre-1.0:

- **minor** (`0.2.0` -> `0.3.0`) -- new capability, or a breaking change. Both,
  because at 0.x there is no separate channel for breaks.
- **patch** (`0.2.0` -> `0.2.1`) -- fixes and internal work that changes no
  interface.

**What would earn 1.0.0**, and none of it has happened yet: the architecture
question settled, so the algorithm is not about to be replaced wholesale; the
cloud-or-not question settled, so modules stop disappearing; and a CLI plus
library surface that someone outside this repository could depend on.

Tag on `main` only, annotated, with a message saying what the release contains:

```bash
git tag -a v0.3.0 -m "..." && git push origin v0.3.0
```

`archive/*` tags are not releases. They preserve retired branch tips and carry no
version meaning.

### The changelog moves with the branch

**`CHANGELOG.md` is updated in the branch that makes the change, not afterwards.**
A branch is not finished until its entry is under `## Unreleased`, so nothing
merges and leaves the changelog to be reconstructed from `git log` later. It had
already drifted to claiming `0.0.1` while the tags said `v0.2.0`; that is what
writing it at merge time prevents.

- [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) headings: **Added**,
  **Changed**, **Deprecated**, **Removed**, **Fixed**, **Security**. Only the
  ones that apply.
- Write what changed for someone using the package, and why when the why is not
  obvious. Not a commit-message copy, and never a bare issue number.
- No entry needed for changes nobody outside the branch can observe -- a `plans/`
  edit, a comment, a pure refactor. Dependency and packaging changes *are*
  observable, so they get one.
- Releasing means renaming `## Unreleased` to `## vX.Y.Z -- YYYY-MM-DD`. That
  section and the annotated tag message say the same thing, so write it once and
  reuse it.

## Conventions

- Cite a plan section by short name plus number: "algorithm design 2.1",
  "conformance audit 5.0". Never a bare number or a file path.
- `plans/` files are living documents. When an item is done, delete it -- no
  strikethrough, no "done" annotations. Git history is the record.
- **`plans/` is the reasoning; GitHub issues are the state.** Why a thing is
  worth doing, and what is known about it, belongs in a plan. Whether it is
  planned, in progress or done belongs in the tracker. Never both -- two
  half-maintained trackers drift apart and then neither can be trusted.
- **Cross-repository work is filed where the code lives.** The rebuild spans
  four repositories -- [POCS](https://github.com/panoptes/POCS) writes the FITS
  headers, `panoptes-utils` is the shared base, this repository processes and
  produces, and
  [panoptes-data](https://github.com/panoptes/panoptes-data) discovers, fetches
  and queries. See data contract 8 for who owns what.
  - A POCS change is a POCS issue. Filing it here instead would put it where
    the people who maintain that code do not look.
  - **The "Photometry rebuild" project board is the single view across all
    four.** It has a Repository field; issues from any of them go on it. The
    board, not a coordinating issue, is what makes the work findable.
  - Reference across repositories fully qualified -- `panoptes/POCS#1410`,
    never a bare `#1410`, which resolves to the wrong repository.
  - An issue here that is blocked by or blocks another repository gets the
    `cross-repo` label and names the other issue in its Dependencies section.
  - The reasoning still lives in `plans/`, in this repository, whichever
    repository the work happens in. Cross-repository issues link back to the
    plan section rather than restating it.
- **Anything needing a human decision is a GitHub issue with the `decision`
  label, under the Decisions milestone, filed immediately.** Not raised in
  conversation and relied on to be remembered. improvement plan 6 was where
  these went before there was a tracker; it is gone.
- **Pull before starting anything.** `git fetch --all --tags && git pull`, then
  read, then plan. `main` and the `v*` tags move between sessions, and a plan
  built on a stale tree proposes work that is already done.
- **`main` is the default and only long-lived branch.** Feature branches are cut
  from it and merged back. `develop` is retired and the triangular workflow is
  gone -- a repository with a `develop` in its history usually still uses it, and
  this one does not. Retired branch tips are preserved as `archive/*` tags
  rather than kept as branches.
- **When you start work on an issue, set its GitHub Project status to "In
  Progress"** -- at the start, not on completion, so the board says what is being
  worked on while it is happening. Closing an issue moves it to Done on its own;
  the starting transition is the one that needs doing.
- **Branch names say what the branch is for**: `type/issue-NNN` plus an optional
  short description, as in `cleanup/issue-170` or
  `cleanup/issue-170-tooling-foundation`. The type is the kind of work
  (`cleanup`, `fix`, `algo`, `docs`); the issue number is what makes the branch
  findable later. A generated name carrying neither -- the
  `claude/adjective-surname-hex` an agent session is handed by default -- is not
  one: rename it before pushing. Name a batch for its parent issue, not for one
  of its children.
- Selection masks are `True == included` everywhere in `lightcurve`. The
  `numpy.ma` convention is the opposite; convert only at that boundary.
- Stamp arrays unpack as `(height, width)`. Stamps are non-square in existing
  data; getting this backwards is conformance audit 5.4.
- **`ruff check .` and `ruff format .` both pass before anything is committed.**
  The repository is clean on both as of #181, so any error you see is one you
  introduced -- there is no pre-existing noise to read past.

  ```bash
  uv run ruff check --fix .   # then read what it could not fix
  uv run ruff format .
  ```

  - Line length 100, double quotes, spaces, LF. Rule set is `E`, `F`, `I`, `UP`:
    pycodestyle errors, pyflakes, import sorting, and pyupgrade.
  - `UP` means modern typing is enforced, not optional: `tuple` not `Tuple`,
    `X | None` not `Optional[X]`. Do not import from `typing` what the builtin
    already provides.
  - **Do not hand-wrap code shorter than the limit.** The formatter joins it
    back and the diff is noise. Let it decide; it wraps what needs wrapping.
  - The formatter cannot split a string, so `E501` inside a docstring or an
    f-string is yours to fix: break it across implicit-concatenated pieces, or
    indent a continuation line in the docstring.
  - A format-only change goes in **its own commit**, never mixed with a real
    one. Reviewing a behavior change through a reflow is how things get missed.
