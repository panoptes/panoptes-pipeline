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

Under active rebuild on the `algorithm-v2` branch. The published description,
[Gee et al., *On-sky Demonstration of Precision Photometry with Bayer Color
Filter Arrays*](https://www.projectpanoptes.org), is earlier research and a
building block rather than a specification.

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

The cloud stack (Firestore, BigQuery, GCS) and the notebook execution path are
extras, so the default environment stays small: `uv sync --extra cloud` or
`uv sync --extra notebooks` when you need them. Standalone scripts under
`scripts/` declare their dependencies inline per PEP 723 and need no sync.

`src/panoptes/pipeline/lightcurve/` holds the algorithm as pure array code —
no I/O, no cloud, no notebook — and is where new work belongs.

## License

MIT. See `LICENSE.txt`.
