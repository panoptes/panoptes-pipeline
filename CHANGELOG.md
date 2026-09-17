# Changelog

Notable changes, newest first, in the [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/) format. The versioning policy
-- and why this project is deliberately pre-1.0 -- is in `AGENTS.md`.

## Unreleased

### Added

- Documentation at <https://panoptes.github.io/panoptes-pipeline/>, built with
  [Zensical](https://zensical.org/) from the docstrings and published to GitHub
  Pages on every push to `main`. A page under `docs/api/` is a `:::` block
  naming a module, so improving a docstring improves the reference and there is
  no second description of the API to keep in step. `plans/` is deliberately not
  published.
- `create-release.yml` publishes a `vX.Y.Z` tag to PyPI through Trusted
  Publishing and creates the GitHub release from the **annotated tag message**,
  which is therefore the one place release notes are written. The workflow
  refuses a lightweight tag rather than publishing a commit subject as the
  notes. `uv build`/`uv publish` by hand is no longer the release process.
- `canary.yml` re-resolves the dependency tree from scratch every Monday and
  runs the suite against it, so an upstream release that breaks this package
  surfaces on a schedule rather than in someone's environment.
- `CONTRIBUTING.md`, `AUTHORS.md`, `CODE_OF_CONDUCT.md` and a
  `.pre-commit-config.yaml` carrying the same ruff settings CI enforces.
  `pre-commit` is in the new `lint` dependency group, so `uv run pre-commit
  install` works from a plain `uv sync`.
- `.gitattributes` marks `CHANGELOG.md` `merge=union`, so parallel branches
  appending under `## Unreleased` stop conflicting.
- `plans/project-standards.md` records how all four PANOPTES repositories are
  built, linted, tested, documented and released, and why each choice is what it
  is.

- `scripts/survey_headers.py` surveys the FITS headers of a local copy of the
  raw archive into `headers.parquet`, one row per frame. It reads headers and
  never pixels, so it needs no processed tree and no plate solving -- which is
  what makes it usable now, and what keeps benchmark selection honest: the
  processed index is built from output of the implementation being replaced.
  WCS written back into archived frames by the retired cloud pipeline is
  excluded for that reason, and kept only as `plate_solved`/`solve_date`
  provenance, since the archive's "raw" frames turn out to be raw to different
  degrees by era. Sharded per sequence, so an interrupted run resumes.
- `panoptes.pipeline.products` writes a frame's metadata document and products
  to `<root>/{unit}/{camera}/{sequence_time}/{image_time}/` as `metadata.json`,
  `image.fits`, `extras.fits` and `sources.parquet`. The pipeline had produced
  no metadata since the Firestore writer was deleted. Documents are validated
  against document-store rules before writing, so attaching a store later is an
  upload rather than a migration.
- `panoptes.pipeline.provenance` tags each calibration value with the tier it
  came from -- `header`, `measured`, `registry` or `default` -- so a measurement
  and a fleet-wide fallback no longer serialize alike. Falling back warns.
- Real POCS frames under `tests/data/`: a raw frame, a solved one, and a header
  carrying almost nothing, plus `widefield.fits.fz` -- a real PANOPTES frame
  rebinned 4x4 so it keeps its 14.9 x 9.9 degree angular size in 0.92 MB. CI now
  installs `astrometry.net` and the Tycho-2 index files and solves it with
  production options, so plate solving is tested rather than skipped.
- `panoptes.pipeline.index` builds the query surface by walking the documents
  the pipeline wrote: `frames.parquet`, one row per frame, and
  `observations.parquet`, one row per sequence derived from it. This replaces
  the Firestore-generated `observations.csv`, which nothing has produced since
  the cloud removal. It reads no pixels, and it computes what the old summary
  could not express -- usable frames as distinct from frames, duration,
  `total_exptime` as a sum over frames rather than a number only the index
  holds, and a flag for a camera uid recording more than one serial across
  sequences. A `schema.json` manifest records the column contract, so a
  consumer in another repository reads a stated schema rather than
  reverse-engineering one.
- `NoSourcesDetected`, raised when nothing in a frame clears the detection
  threshold. Clouds, a closed dome and lost tracking all produce such a frame,
  so it is an ordinary outcome over a survey and callers can distinguish it from
  a genuine failure.
- `scripts/fetch_catalog.py` builds a local catalog for a field from a Gaia DR3
  cone search. Nothing in this repository could previously produce the catalog
  the pipeline requires -- the archive's catalog came from a BigQuery table that
  was deleted with the cloud stack -- so the pipeline could not be run on a frame
  at all. `picid`
  is exactly the Gaia DR3 `source_id`, so no crossmatch is involved. PEP 723, so
  `astroquery` stays out of the package. The default filename states the query
  that produced it, and an existing, usable file is left alone, so re-running
  for a field already fetched costs nothing.
- `panoptes.pipeline.worklist` classifies every raw frame against an output
  root as missing, params changed, prior error, incomplete, forced or up to
  date, without opening any pixels. `as_table` renders the result as a CSV-able
  table, so a settings change can be shown to have invalidated what was expected
  and nothing more.
- `PipelineParams.fingerprint`, a short digest of every parameter that affects a
  product. Settings were previously stored in each document but never compared,
  so a change left stale products in place with no signal.
- `panoptes.pipeline.status` holds `ImageStatus` and `ObservationStatus`, moved
  back from `panoptes-data`. They describe this pipeline's own stages, so the
  client that reads its output should not own them.
- `products.record_processing` stamps a document with the settings, fingerprint
  and stage it was produced at; `products.read_document` reads one back,
  returning None rather than raising on a file that will be rewritten anyway.
- `panoptes.pipeline.processing` restores the two entry points as library
  functions: `process_frame` for a single FITS, `process_observation` for a
  sequence. Calibration is `calibrate`; the sequence document is written once
  at the end of a batch, by the only writer that exists at that point.

### Changed

- Lint settings are now shared with `panoptes-utils` and POCS: 110 columns and
  `E`, `W`, `F`, `I`, `UP`. The accompanying reflow touched 24 files and changed
  no behavior.
- Tooling dependencies are split into `lint`, `test` and `docs` groups, with
  `dev` including all three. `uv sync` with no arguments is unchanged.
- `AGENTS.md` is the canonical agent instruction file; `CLAUDE.md` and
  `GEMINI.md` are symlinks to it.
- `tests.yml` runs lint as its own job, installs with `uv sync --locked` so a
  stale lockfile fails CI rather than being silently re-resolved, and uploads
  coverage to Codecov.
- `panoptes.pipeline.utils` is a regular package rather than an implicit
  namespace package. Imports are unaffected; static readers, including the one
  that builds the API reference, can now see the four modules in it.
- `scikit-image` is a declared dependency again. v0.4.0 dropped it from the
  retired `env.yaml` on the grounds that nothing imported it, which was true
  of this package's own source and missed that `photutils` imports it at
  runtime for deblending. A plain `uv sync` therefore produced an environment
  where source detection raised `ModuleNotFoundError` -- the split environment
  that release set out to eliminate. Checking direct imports is not enough to
  retire a dependency.
- `extract_metadata` reads image dimensions from `NAXIS1`/`NAXIS2` rather than
  `IMAGEW`/`IMAGEH`, which only plate solving writes -- raw frames silently
  yielded zero.
- The camera id and body serial are now in the `image` document as well as
  `sequence`, so a per-frame measurement can be joined to its camera.
- `extract_metadata` takes an optional `CameraSettings` for header fallbacks.
- A frame's pixels are one multi-extension `image.fits` -- `PRIMARY` reduced,
  plus `BACKGROUND`, `RMS` and `MASK` -- instead of `image.fits` and
  `extras.fits`. `extras_filename` is gone. Raw pixels are no longer copied
  into the processed tree, and the background is stored as the low-resolution
  mesh photutils actually fits rather than its interpolation, which is four
  orders of magnitude larger. Roughly 100 MB per frame instead of 312 MB.
- `CameraSettings.saturation` is now raw ADU, matching `WHTLVLN`, and
  saturation is masked before bias subtraction. The default moves `15872` ->
  `16384`; those are the same number in different domains (`15872 = 2**14 -
  512`), but mixing them would have over-masked by the bias on every frame.
- `plate_solve` no longer replaces the file it solves, and honours its own
  `timeout` argument instead of a hardcoded 300 seconds.
- `match_sources` takes the frame's dimensions explicitly. They previously came
  from fleet-wide settings that the caller was expected to overwrite per frame,
  which would now change the params fingerprint from frame to frame.
- `panoptes-utils` floor raised to `0.3.1`, which introduced `ImagePathInfo`;
  it is now imported from `panoptes.utils.images.fits`.

### Fixed

- `sources.get_catalog_match` documents the function that exists. Its docstring
  described a `return_unmatched` argument -- in `Args:`, a `Note:`, a `Warning:`
  and a paragraph of prose -- and an `origin` argument twice, neither of which is
  in the signature. Passing either went into `**kwargs` and was silently ignored,
  so a caller asking for unmatched catalog entries got matched ones and no error.
  The appending of unmatched entries is not implemented; the docstring now says
  what the function returns instead of promising a switch for it. It also no
  longer calls `wcs` optional -- `get_xy_positions` needs it on every call, so
  passing `catalog_stars` without a `wcs` raises `AttributeError` after the
  matching is done -- and no longer points at a `lookup_point_sources` that does
  not exist, or imply the stale `measured_ra`/`measured_dec` defaults are what
  `detect_sources` produces.
- `sources.get_catalog_match` and `sources.get_stars` no longer claim the catalog
  is derived from the TESS Input Catalog v8. `picid` is exactly the Gaia DR3
  `source_id`, so a catalog is a Gaia cone search with its columns renamed and no
  crossmatch step -- which is what `scripts/fetch_catalog.py` builds. The listed
  added columns were also wrong: the function adds `catalog_sep` and the four
  `catalog_wcs_{x,y}[_int]` columns, not `catalog_sep_arcsec`, `catalog_x`,
  `twomass` or the `catalog_measured_diff_*` family.

- `process_frame` reports the reduced image it wrote. The path was returned by
  `write_image` and then dropped, so the result named every product except the
  largest one.
- A frame whose header has no `FILENAME` keyword can be identified. `worklist.identify`
  fell through to `ImagePathInfo.from_fits_header`, which reads that keyword
  first and catches only `ValueError`, so a missing one raised `KeyError` and
  `process_frame` could not place the frame at all.

- A catalog carrying only the four documented columns now works. `match_sources`
  read `catalog_gaiabp`, `catalog_gaiarp` and `catalog_gaiamag` unconditionally,
  so a catalog built to the stated contract passed validation, matched, and then
  raised `AttributeError` -- after plate solving and source detection had already
  run. Those three are optional enrichment: present, three color-excess columns
  are derived; absent, they are skipped. Nothing reads the excesses yet.

- An absent `CAMSN` records as null instead of the string `"None"`.
- A frame's metadata document is written after its products, not before. It is
  the marker the work-list walk reads to decide a frame is done, so a failure
  part way through the products used to leave a frame claiming to be finished
  with its products missing, and the next walk skipped it.
- `process_observation` only requires `solve-field` when the work list is not
  empty, so re-running to confirm there is nothing to do no longer fails on a
  machine without a solver.
- Source detection could not complete on any frame at all. A masked array was
  handed to `photutils` as `data` while its mask was also passed separately;
  `photutils` computes image moments as a matrix product and `numpy.ma` cannot
  propagate a mask through `@`, so `SourceCatalog` raised "operands could not be
  broadcast together". Detection now returns sources on a real frame.
- A frame with no detections raises `NoSourcesDetected`, naming the threshold it
  failed to clear, instead of `TypeError: segmentation_image must be a
  SegmentationImage` from inside `photutils` -- which said nothing about the
  frame.
- Source detection no longer crashes on a frame with saturated pixels. Its mask
  guard tested a masked array for truthiness, which raises for any real mask and
  built a 0-d mask when there was none. It was unreachable only because the
  fleet-wide saturation default was too high to mask anything; resolving
  saturation from the header makes real masks appear.

### Removed

- `panoptes-data` is no longer a dependency: nothing imported it once
  `ImagePathInfo` moved to `panoptes-utils`, and it is downstream of this
  pipeline. The `<0.2` pin goes with it, along with 21 transitive packages
  including `ipython` and `ipywidgets`.

## v0.4.0 -- 2026-09-12

The package stops requiring Google Cloud, Docker, conda and Jupyter. Every
module under `src/panoptes/pipeline/` imports from a plain `uv sync`, and the
algorithm runs against a local directory with no credentials and no cloud
project -- which is what a citizen-science pipeline needs in order for a
contributor to run it at all.

The cost is stated plainly: this discards working archive integration and the
only existing path from raw frames to products. Both were producing the
un-subtracted stamps the rebuild is trying to get away from, and git history
keeps every file.

### Added

- `sources.read_catalog`, which reads the catalog as **parquet, ECSV or CSV**,
  chosen by suffix (`.parquet`, `.pq`, `.ecsv`, `.csv`, `.tsv`, each optionally
  compressed). Parquet is the better default for an all-sky catalog; astropy's
  ECSV is the one to reach for when the file should stay readable, since its
  YAML header carries the column types. Plain CSV and TSV are accepted so an
  existing catalog needs no conversion step.
- `picid` is checked as an integer on read and **returned as a categorical** --
  it names a star rather than measuring one. Filtering prunes unused categories,
  so a field cut from an all-sky catalog does not carry every id in the sky.
  Note when consuming it: `groupby` on a categorical iterates every category
  unless passed `observed=True`. The integer check is what catches plain CSV's
  missing dtypes, where a single blank turns that column into floats that join
  as `1234.0` against integer ids and silently match nothing.
- `tests/test_sources_catalog.py`, pinning the catalog filtering semantics the
  BigQuery query used to own: a half-open `[vmag_min, vmag_max)` range, an
  inclusive positional box, and a Right Ascension window that may wrap through
  zero. The filtering tests run against every accepted format, since the format
  must not change the answer. 139 tests, up from 78.

### Changed

- **`sources.get_stars` reads a local catalog file instead of querying
  BigQuery.** Callers pass `catalog_filename`, or set
  `params.catalog.catalog_filename`. There is no network lookup and no fallback,
  so an unset path raises with a message naming the setting rather than silently
  reaching for credentials. The file must carry the mapped PIC column names --
  `picid`, `catalog_ra`, `catalog_dec`, `catalog_vmag` -- and one missing any of
  them raises and names them. Those four are the entire contract, so the catalog
  can be rebuilt without touching this package.
- **`images.match_sources` now filters the local catalog** by the WCS footprint
  and the configured Vmag limits. It previously read the whole file unfiltered
  whenever a local catalog was set, applying those bounds only on the BigQuery
  path -- so `vmag_limits` was silently ignored offline, and a full-sky catalog
  was matched in its entirety rather than over the field. The two routes now do
  the same thing.
- The `bq_client`, `bqstorage_client`, `column_mapping` and `return_dataframe`
  arguments are gone from `get_stars`; `catalog_filename` replaces them.
- Coverage configuration moved from `.coveragerc` into `[tool.coverage.*]` in
  `pyproject.toml`, alongside the ruff and pytest configuration. Settings are
  unchanged.
- `README.md` rewritten: it documented `uv sync --extra cloud` and
  `--extra notebooks`, neither of which exists, so both commands failed. It now
  describes the install as it is, the catalog file source matching needs, and
  how to contribute -- branch from `main`, name the branch for its issue, run
  pytest and ruff, move the changelog in the same branch.
- Two docstrings in `utils/sources.py` described matched columns as coming from
  source-extractor. They come from photutils.

### Removed

- **The papermill execution path, in full**: both tracked notebooks, the
  `panoptes-pipeline` console script and the `utils/cli` package behind it,
  `utils/notebooks.py`, `image.py`, `observation.py`, and the FastAPI service in
  `services/processing.py`. The notebooks were the implementation rather than a
  demonstration of it, and everything above existed only to run them, so with
  the notebooks gone none of it could execute. **There is no path from raw
  frames to products** until the library extraction and the new CLI land.
- **Google Cloud, in full**: `utils/gcp/` (BigQuery, Firestore and GCS helpers),
  the eight cloud packages, and `.gcloudignore`. Removal was chosen over putting
  cloud behind adapters, so there is no cloud code path at all rather than one
  that is off by default -- an offline path that is the only path cannot rot
  quietly while the cloud path is the one being exercised.
- **Docker and conda, in full**: the `Dockerfile`, `.dockerignore` and
  `env.yaml`. The image existed to run the FastAPI service, so its entrypoint
  pointed at a module that no longer exists; it would have built and then failed
  to start. `env.yaml` was that image's conda environment and its only consumer,
  duplicating `pyproject.toml` imperfectly -- `astroplan`, `db-dtypes`,
  `jupyterlab_execute_time` and `scikit-image` were listed there and nowhere
  else, and nothing imports any of them. The image installed conda, then mamba,
  then pip on top of both. `uv sync` is the supported install; there is no
  container build.
- **All optional-dependency extras.** The `cloud` and `notebooks` extras went
  with the code they installed, so there is no longer an environment in which
  part of the package works and part raises `ModuleNotFoundError`.
- **The `docs/` tree, `.readthedocs.yml` and the `docs` dependency group**, plus
  the `Documentation` project URL. `https://panoptes-pipeline.readthedocs.io`
  returns 404 -- the project does not exist on Read the Docs -- and what was not
  being published was generator boilerplate whose landing page still opened with
  "This is the main page of your project's Sphinx documentation". `README.md`
  and `plans/` are the documentation.
- `resources/source-extractor/`. Nothing referenced it; detection runs through
  `photutils.segmentation`.
- `AUTHORS.md` and `CONTRIBUTING.md`, folded into `README.md`. The latter was a
  single line pointing at another repository's `develop` branch, describing a
  workflow this project retired.

## v0.3.0 -- 2026-09-12

### Added

- `uv.lock`, pinning 238 packages. Precision numbers are compared across weeks,
  so an unpinned scipy changing a solver default underneath a benchmark is a
  reproducibility problem, not a convenience one.
- `cloud` and `notebooks` optional dependency extras, holding the legacy GCP
  orchestration and the papermill execution path. Neither is installed by
  default; both are slated for removal.
- `dev` and `docs` dependency groups (PEP 735) for test, lint and docs tooling.
- `tests/__init__.py`, so `from tests.synthetic import ...` resolves under plain
  `pytest` rather than only under `python -m pytest`.

### Changed

- `uv sync` produces an environment where the package imports, so the documented
  test command is `uv run pytest` -- no `PYTHONPATH=src`, no `--with` list.
- `pydantic-settings`, `typer`, `matplotlib`, `astropy`, `tqdm` and
  `python-dateutil` are declared dependencies instead of arriving transitively.
- `panoptes-data` pinned `<0.2`, which dropped `panoptes.data.images`. A holding
  action; see issue #176.
- `google-cloud-bigquery-storage[pandas]` relaxed from `==2.6.2` to `>=2.6.2`.
- The `Programming Language :: Python :: 3.8` classifier is now `3.12`, matching
  `requires-python`.
- Read the Docs builds on Python 3.12; it asked for 3.11, which cannot install a
  package requiring `>=3.12`.

### Removed

- All six `[tool.hatch.envs.*]` blocks. `hatchling` and `hatch-vcs` remain the
  build backend and version source; only the task-runner layer is gone, replaced
  by documented `uv` invocations including `uv build` and `uv publish`.
- The unreachable `importlib-metadata; python_version < '3.8'` marker and its
  counterpart, the dead `sys.version_info >= (3, 8)` branch in
  `src/panoptes/pipeline/__init__.py`.
- The `testing` extra, folded into the `dev` dependency group; tooling is not
  package metadata.
- `mocket`, which nothing imports.

## v0.2.0 -- 2026-09-12

The rebuild's starting line.

First tag since v0.1.0 (January 2020, 166 commits back). Marks the point where
the repository was cleaned up and the algorithm rebuild became the work: a
pure-numpy lightcurve package with no I/O or cloud dependency, the benchmark
harness, the plan documents, a single main branch, and a GitHub tracker with the
work broken out.

Pre-1.0 deliberately. The cloud modules are being deleted, the CLI is being
rewritten, and the core algorithm is pending a head-to-head test, so the public
API is not stable and should not be treated as such.

## v0.1.0 -- 2020-01-19

WCS footprint improvements.
