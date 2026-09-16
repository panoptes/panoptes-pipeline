# Changelog

Notable changes, newest first, in the [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/) format. The versioning policy
-- and why this project is deliberately pre-1.0 -- is in `CLAUDE.md`.

## Unreleased

### Added

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
  carrying almost nothing.
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

### Changed

- `extract_metadata` reads image dimensions from `NAXIS1`/`NAXIS2` rather than
  `IMAGEW`/`IMAGEH`, which only plate solving writes -- raw frames silently
  yielded zero.
- The camera id and body serial are now in the `image` document as well as
  `sequence`, so a per-frame measurement can be joined to its camera.
- `extract_metadata` takes an optional `CameraSettings` for header fallbacks.
- `panoptes-utils` floor raised to `0.3.1`, which introduced `ImagePathInfo`;
  it is now imported from `panoptes.utils.images.fits`.

### Fixed

- An absent `CAMSN` records as null instead of the string `"None"`.

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
