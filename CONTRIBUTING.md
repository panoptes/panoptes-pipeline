# Contributing

Thanks for helping with `panoptes-pipeline`.

This package does differential photometry: it takes raw PANOPTES frames and
produces lightcurves precise enough to recover a transit. The governing document
is [`plans/algorithm-design.md`][design] -- what the algorithm is, stated
independently of any implementation. Read it before changing anything under
`src/panoptes/pipeline/lightcurve/`. The published paper (Gee et al., *On-sky
Demonstration of Precision Photometry with Bayer Color Filter Arrays*) is a
building block, not a specification; where its choices and the idea diverge, the
idea wins.

[design]: plans/algorithm-design.md

## Issue reports

Please check the [issue tracker] first, including closed issues. When reporting
a bug, include the version (`pip show panoptes-pipeline`), your operating system
and Python version, and the steps to reproduce it. A frame that triggers it is
worth more than a description of one.

The rebuild spans four repositories, and an issue belongs where the code lives:
[POCS] writes the FITS headers, [`panoptes-utils`][utils] is the shared base,
this repository processes and produces, and [`panoptes-data`][data] discovers,
fetches and queries. Filing a header bug here rather than in POCS puts it where
the people who maintain that code do not look.

Cross-repository references are written fully qualified --
`panoptes/POCS#1410`, never a bare `#1410` -- because a bare number resolves
against whichever repository the reader happens to be looking at.

[POCS]: https://github.com/panoptes/POCS
[utils]: https://github.com/panoptes/panoptes-utils
[data]: https://github.com/panoptes/panoptes-data

## Getting set up

Everything goes through [uv]. `requires-python` is `>=3.12`, and `uv` resolves a
matching interpreter on its own, so there is no virtual environment to create by
hand:

```bash
git clone git@github.com:panoptes/panoptes-pipeline.git
cd panoptes-pipeline
uv sync
```

`uv sync` with no arguments installs the project plus the `dev` group, so the
package is importable and no `PYTHONPATH` is needed. There are no extras: every
module under `src/panoptes/pipeline/` imports from a plain `uv sync`, and there
is no environment in which part of the package works and part raises
`ModuleNotFoundError`.

```bash
uv run pytest
```

The suite needs no credentials and no network. Plate solving runs for real
against a rebinned frame committed under `tests/data/`, which needs
`astrometry.net` and its index files; without them those tests skip and say so.
On Debian or Ubuntu:

```bash
sudo apt-get install -y astrometry.net astrometry-data-tycho2-10-19 libcfitsio-bin
```

`libcfitsio-bin` supplies `funpack`. Without it a `.fz` frame cannot be
unpacked, and every solve of a compressed frame fails with "no WCS header
present" -- while uncompressed frames still solve, which makes it look like a
fixture problem rather than a missing tool.

Standalone scripts under `scripts/` are [PEP 723]: they declare their own
dependencies inline and run against a throwaway environment rather than the
synced one.

```bash
uv run scripts/fetch_catalog.py 86.49 8.72
```

That one is how you get a catalog at all. `params.catalog.catalog_filename`
names a local file and there is no network lookup, so without it the pipeline
reaches catalog matching and stops.

## Making a change

**Pull before you start.** `git fetch --all --tags && git pull`. `main` and the
`v*` tags move between sessions, and a plan built on a stale tree proposes work
that is already done.

**Branch from `main`** -- it is the only long-lived branch -- and name the branch
for the work: `type/issue-NNN` plus an optional short description, as in
`fix/issue-170` or `algo/issue-171-drift`. The type is the kind of work (`fix`,
`cleanup`, `algo`, `docs`, `infra`).

**Lint and format before you commit.** The rule set and style are pinned in
`pyproject.toml` and shared with the other three repositories:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Both run in CI as their own job. A format-only change goes in its own commit,
never mixed with a behavior change: reviewing a real change through a reflow is
how things get missed.

Optionally, install the hooks so this happens without being remembered:

```bash
uv run pre-commit install
```

**Update `CHANGELOG.md` in the same branch**, under `## Unreleased`, using
[Keep a Changelog] headings. A branch is not finished until its entry is there.
No entry is needed for changes nobody outside the branch can observe -- a
`plans/` edit, a comment, a pure refactor -- but dependency and packaging
changes are observable, so they get one.

**Never hardcode a version.** `hatch-vcs` derives it from the nearest `v*` tag,
which is the only source of truth.

**Never hardcode a camera constant.** The fleet is heterogeneous: many units,
many DSLR bodies, different black levels, white levels, gain, sensor and pixel
sizes. Measure it -- saturation from the pixel histogram, gain by photon
transfer, pixel scale from the WCS, Bayer phase via `masks.infer_pattern` -- or
resolve it from a profile keyed on the `CAMSN` serial, and **fail loudly rather
than defaulting**. Two bugs in this repository came from fleet-wide defaults.

**Reach for [`panoptes-utils`][utils] first.** It is the shared base across all
four repositories and this package already imports it in several modules.
Before writing a helper, check whether it is there; a local reimplementation is
not merely duplicate, it is a *second* answer to a question that already has
one, and the two drift. If the shared helper is close but wrong, fix it there.

American English throughout: code, comments, docstrings, commits, issues and the
changelog.

## Tests

**A test for a fixed bug is run against the unfixed code first, and the commit
says so.** Stash the source, run the new test, watch it fail, restore:

```bash
git stash push src/panoptes/pipeline/<changed>.py   # then run the new test
```

This is not ceremony. Three tests in one batch here passed while asserting
nothing, and line coverage found none of them -- they all executed the code
under test and the suite was green. Where there is no "before", break the code
by hand instead: invert the condition, return a constant, delete the line.

Two habits catch the same class more cheaply:

- **The object of the action and the object of the assertion must be the same
  variable.** `plate_solve(filename=widefield)` followed by an assertion about
  `products_dir` cannot fail.
- **When the setup builds a condition, assert the condition before asserting the
  behavior.** If the setup silently did not take, the test is measuring the
  default path.

## Measuring an algorithm change

Numbers about precision are held to a higher standard than the rest, because the
ways of getting a flattering one are well known and easy to reach by accident.
`plans/improvement-plan.md` has the full argument; the short version:

- **Report transfer alongside noise, always.** `injection.transfer_function`
  injects sinusoids across a range of periods and reports recovered amplitude
  over injected. A change that lowers scatter while lowering transfer has
  suppressed signal, not removed noise.
- **Inject at the pixel level**, before normalization, so the whole pipeline
  sees the signal. Injecting into a finished lightcurve measures nothing.
- **Score on held-out frames.** Fitting many free parameters per target will
  absorb noise given the chance.
- **Report the per-target win rate** across a sample, not one target's median. A
  single flattering target is how the first baseline here came out 10x too
  optimistic.

Correcting a defect can make numbers look worse. Removing a bias toward flat
lightcurves is a real improvement presenting as a regression; say so rather than
tuning it away.

Note also that a lightcurve is **never normalized to unit median**. The survey
targets long-period planets, and for a window wholly or partly in transit that
subtracts the signal from itself.

## Submitting your contribution

Push your branch and open a pull request against `main`. Say what changed and
why; the diff shows the what, so the why is the part only you have.

CI runs lint, the test suite on every supported Python version, and a
documentation build. Please get them green -- or say what you are stuck on,
which is a perfectly good reason to open the pull request as a draft.

## Maintainer tasks

### Releases

Releases are cut by tagging, and only when explicitly intended:

1. Make sure CI is green on `main` and `CHANGELOG.md` describes the release.
2. Rename `## Unreleased` to `## vX.Y.Z -- YYYY-MM-DD`.
3. Create an **annotated** tag `vX.Y.Z` on `main`. Its message becomes the
   GitHub release notes, so what the tag says and what the release says cannot
   drift apart -- write it once and reuse the changelog section.
4. Push the tag.

`create-release.yml` then builds the package, publishes it to PyPI through
Trusted Publishing, and creates the GitHub release from the tag message.

This project is pre-1.0 deliberately. While pre-1.0, **minor** covers new
capability or a breaking change -- both, because at `0.x` there is no separate
channel for breaks -- and **patch** covers fixes and internal work that changes
no interface. `archive/*` tags are not releases; they preserve retired branch
tips and carry no version meaning.

[uv]: https://docs.astral.sh/uv/
[PEP 723]: https://peps.python.org/pep-0723/
[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[issue tracker]: https://github.com/panoptes/panoptes-pipeline/issues
