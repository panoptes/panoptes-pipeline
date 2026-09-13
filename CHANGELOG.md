# Changelog

Notable changes, newest first, in the [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/) format. The versioning policy
-- and why this project is deliberately pre-1.0 -- is in `CLAUDE.md`.

## Unreleased

### Removed

- The papermill execution path, in full: both tracked notebooks, the
  `panoptes-pipeline` console script and the `utils/cli` package behind it,
  `utils/notebooks.py`, `image.py`, `observation.py`, and the FastAPI service in
  `services/processing.py`. The notebooks were the implementation rather than a
  demonstration of it, and everything above existed only to run them, so with
  the notebooks gone all of it was code that could not execute. There is no path
  from raw frames to products until the library extraction and the new CLI land;
  the path that existed produced the un-subtracted stamps the rebuild is trying
  to get away from.
- The `notebooks` extra, with the execution path it installed. `fastapi`,
  `gunicorn`, `uvicorn` and `google-cloud-pubsub` leave the `cloud` extra with
  the service; the remaining cloud clients stay, because `utils/gcp`,
  `utils/sources` and `utils/images` still import them.
- The `docs/` tree, `.readthedocs.yml`, the `docs` dependency group and the
  `Documentation` project URL. `https://panoptes-pipeline.readthedocs.io`
  returns 404 -- the project does not exist on Read the Docs -- and what was
  being not-published was generator boilerplate whose landing page still opened
  with "This is the main page of your project's Sphinx documentation".
- `resources/source-extractor/`. Nothing referenced it; detection runs through
  `photutils.segmentation`.
- `AUTHORS.md` and `CONTRIBUTING.md`, folded into `README.md`. The latter was one
  line pointing at another repository's `develop` branch, describing a workflow
  this project retired.
- `.coveragerc`, migrated to `[tool.coverage.*]` in `pyproject.toml` alongside
  the ruff and pytest configuration. Settings are unchanged.
- **Google Cloud, in full**: `utils/gcp/` (BigQuery, Firestore and GCS
  helpers), the `cloud` optional-dependency extra and its eight packages, and
  `.gcloudignore`. Removal was chosen over putting cloud behind adapters, so
  there is no cloud code path at all rather than one that is off by default.
  Consequence worth knowing: **every module under `src/panoptes/pipeline/` now
  imports from a plain `uv sync`.** There is no longer an environment in which
  part of the package works and part raises `ModuleNotFoundError`.
- **Docker, in full**: the `Dockerfile`, `.dockerignore` and `env.yaml`. The
  image existed to run the FastAPI service, which is deleted and is not being
  replaced, so its entrypoint pointed at a module that no longer exists -- it
  would have built and then failed to start. There is no container build for
  this project; `uv sync` is the supported install.
- `env.yaml` went with it, being a conda environment whose only consumer was
  that image. It duplicated `pyproject.toml` imperfectly and had drifted:
  `astroplan`, `db-dtypes`, `jupyterlab_execute_time` and `scikit-image` were
  listed there and nowhere else. Nothing imports any of them, so nothing moved
  across. The image installed conda, then mamba, then pip on top; all three are
  gone.

### Added

- `sources.read_catalog`, which reads the catalog as **parquet, ECSV or CSV**,
  chosen by suffix (`.parquet`, `.pq`, `.ecsv`, `.csv`, `.tsv`, each optionally
  compressed). Parquet remains the better default for an all-sky catalog;
  astropy's ECSV is the one to reach for when the file should stay readable,
  since its YAML header carries the column types. Plain CSV and TSV are accepted
  so an existing catalog needs no conversion step.
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
  must not change the answer. 139 tests.

### Changed

- **`sources.get_stars` reads a local catalog file instead of querying
  BigQuery.** Callers pass `catalog_filename` (or set
  `params.catalog.catalog_filename`); there is no network lookup and no
  fallback, so an unset path raises with a message naming the setting rather
  than silently reaching for credentials. The file must carry the mapped PIC
  column names -- `picid`, `catalog_ra`, `catalog_dec`, `catalog_vmag` -- and a
  file missing any of them raises and names them.
- **`images.match_sources` now filters the local catalog** by the WCS footprint
  and the configured Vmag limits. It previously read the whole parquet
  unfiltered when a local catalog was set, and only applied those bounds on the
  BigQuery path, so the two routes disagreed. They now do the same thing.
- The `bq_client`, `bqstorage_client`, `column_mapping` and `return_dataframe`
  arguments are gone from `get_stars`; `catalog_filename` replaces them.
- `README.md` now carries contributing instructions and says the rebuild happens
  on `main`. It claimed an `algorithm-v2` branch that no longer exists.
- Two docstrings in `utils/sources.py` described matched columns as coming from
  source-extractor. They come from photutils.


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
