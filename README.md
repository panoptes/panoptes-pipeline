# panoptes-pipeline

Differential photometry for [Project PANOPTES](https://www.projectpanoptes.org):
recovering exoplanet transits from DSLR wide-field images with a Bayer color
filter array.

## The idea

A wide field gives any target many candidate reference stars whose light lands
on the color filter array the same way the target's does. Those references
build an idealized star that behaves as the target would absent a transit, so
the difference is zero unless the target's flux genuinely changed.

Dividing a stamp by its own summed flux marginalizes brightness out and leaves
the spatial profile, which is what separates shape from brightness: match on
shape, difference the brightness.

The algorithm and the network are the same principle at two scales, and low unit
cost enables both — many stars per unit to calibrate the systematic, many units
at different longitudes to cover a transit longer than one night.

## Status

Under active rebuild on `main`, which is the only long-lived branch. The
published description, [Gee et al., *On-sky Demonstration of Precision
Photometry with Bayer Color Filter Arrays*](https://www.projectpanoptes.org),
is earlier research and a building block rather than a specification.

What is planned, in progress and done lives in [GitHub
issues](https://github.com/panoptes/panoptes-pipeline/issues), grouped into
milestones. The plan documents below carry the reasoning behind that work.

Start with the design documents:

- **`plans/algorithm-design.md`** — what the algorithm is, independent of any
  implementation, and the proposed architecture. Its section 5 says which
  supporting measurements are solid and which are provisional.
- `plans/conformance-audit.md` — defects in the existing implementation.
  Historical context for the rebuild, not a task list.
- `plans/improvement-plan.md` — metrics, benchmark selection, fleet
  heterogeneity, and the open decisions in its section 6.

`CLAUDE.md` carries the same orientation for agent sessions.

## Install and test

Everything goes through [`uv`](https://docs.astral.sh/uv/), against the
committed `uv.lock`:

```shell
uv sync         # project + dev tooling, pinned
uv run pytest
```

That is the whole install. The pipeline runs against a local directory of FITS
files — no cloud project, no credentials. Standalone scripts under `scripts/`
declare their dependencies inline per PEP 723 and need no sync.

`src/panoptes/pipeline/lightcurve/` holds the algorithm as pure array code —
no I/O, no cloud, no notebook — and is where new work belongs.

Source matching needs a local copy of the PANOPTES Input Catalog, pointed at by
`params.catalog.catalog_filename`. Parquet, ECSV or CSV, chosen by suffix; it
must carry `picid`, `catalog_ra`, `catalog_dec` and `catalog_vmag`.

## Contributing

Contributions are welcome — this is a citizen-science project and the pipeline
is being rebuilt in the open.

- **Start from an issue.** If there is not one for what you want to do, open
  one first. Anything needing a decision rather than an implementation gets the
  `decision` label.
- **Branch from `main`,** which is the only long-lived branch, and merge back
  into it. Name the branch for what it is for: `type/issue-NNN`, as in
  `cleanup/issue-171` or `fix/issue-93-background`.
- **Run the checks before opening a pull request:**

  ```shell
  uv run pytest
  uv run ruff check .
  uv run ruff format .
  ```

- **Update `CHANGELOG.md` in the same branch,** under `## Unreleased`, for any
  change someone using the package could observe. A branch is not finished
  until its entry is written.
- **Precision claims need evidence.** A change that lowers scatter while
  lowering signal transfer has suppressed signal rather than removed noise, so
  report both. `CLAUDE.md` has the full rules under "Measuring a change".

Questions are welcome on the [PANOPTES
forum](https://forum.projectpanoptes.org).

## Credits

The PANOPTES Team, <developers@projectpanoptes.org>, and the [contributors to
this repository](https://github.com/panoptes/panoptes-pipeline/graphs/contributors).

## License

MIT. See `LICENSE.txt`.
