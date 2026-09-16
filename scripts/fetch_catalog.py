#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "astropy",
#   "astroquery",
#   "pandas",
#   "pyarrow",
#   "typer",
# ]
# ///
"""Build a local catalog for one field, straight from Gaia DR3.

The pipeline has no network catalog lookup -- `params.catalog.catalog_filename`
names a local file and `get_stars` fails loudly when it is unset. That is
deliberate, but it leaves no way to run the pipeline at all without a catalog
from somewhere, and the archive's own catalog came out of a BigQuery table that
no longer has a code path here.

This closes that gap with a cone search. **`picid` is exactly the Gaia DR3
`source_id`**, an alias and nothing more, so the catalog the pipeline wants is a
Gaia query with its columns renamed -- no crossmatch, no intermediate service.

    uv run scripts/fetch_catalog.py 86.49 8.72 --radius 9 --vmag-max 13 -o pic.parquet

The output is what `CatalogSettings.catalog_filename` expects: parquet, ECSV or
CSV by suffix. See data contract 6 for where camera-level values come from; this
is the sky side of the same problem.
"""

from __future__ import annotations

from pathlib import Path

import typer

#: Gaia columns, and the pipeline names they map onto. `phot_g_mean_mag` stands
#: in for `catalog_vmag`: Gaia G is not Johnson V, but the catalog uses it as
#: the brightness on which `vmag_limits` selects, and the archive's own catalog
#: does the same. Anything needing true V has to say so.
COLUMN_MAP = {
    "source_id": "picid",
    "ra": "catalog_ra",
    "dec": "catalog_dec",
    "phot_g_mean_mag": "catalog_vmag",
    "phot_bp_mean_mag": "catalog_gaiabp",
    "phot_rp_mean_mag": "catalog_gaiarp",
    "phot_g_mean_mag_dup": "catalog_gaiamag",
}

app = typer.Typer(add_completion=False)


@app.command()
def main(
    ra: float = typer.Argument(..., help="Field centre RA, degrees."),
    dec: float = typer.Argument(..., help="Field centre Dec, degrees."),
    radius: float = typer.Option(9.0, help="Cone radius, degrees."),
    vmag_min: float = typer.Option(6.0, help="Brightest Gaia G to include."),
    vmag_max: float = typer.Option(13.0, help="Faintest Gaia G to include."),
    output: Path = typer.Option(Path("catalog.parquet"), "-o", "--output"),
    limit: int = typer.Option(-1, help="Row cap; -1 for no cap."),
) -> None:
    """Fetch a Gaia DR3 cone and write it as a pipeline catalog."""
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = limit
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    # A cone big enough to cover the frame's corners, filtered on G so the
    # download is the stars the pipeline would actually extract. Sources with no
    # BP or RP are dropped: the colour-excess columns would be null and
    # reference selection cannot use them.
    query = f"""
        SELECT source_id, ra, dec,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(POINT({ra}, {dec}), CIRCLE(ra, dec, {radius}))
          AND phot_g_mean_mag BETWEEN {vmag_min} AND {vmag_max}
          AND phot_bp_mean_mag IS NOT NULL
          AND phot_rp_mean_mag IS NOT NULL
    """
    typer.echo(f"Querying Gaia DR3 around ({ra}, {dec}) within {radius} deg ...")
    table = Gaia.launch_job_async(query).get_results()
    typer.echo(f"Got {len(table)} sources.")

    frame = table.to_pandas().rename(
        columns={k: v for k, v in COLUMN_MAP.items() if k != "phot_g_mean_mag_dup"}
    )
    # `catalog_gaiamag` is Gaia G, the same quantity `catalog_vmag` carries. It
    # is duplicated rather than aliased because the two are used for different
    # things -- brightness selection and colour excess -- and a later change to
    # either should not silently move the other.
    frame["catalog_gaiamag"] = frame["catalog_vmag"]

    frame["picid"] = frame["picid"].astype("int64")
    frame = frame[list(dict.fromkeys(COLUMN_MAP.values()))]

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        frame.to_parquet(output, index=False)
    elif output.suffix == ".csv":
        frame.to_csv(output, index=False)
    else:
        from astropy.table import Table

        Table.from_pandas(frame).write(output, overwrite=True)

    size = output.stat().st_size / 1e6
    typer.echo(f"Wrote {len(frame)} rows to {output} ({size:.1f} MB)")
    typer.echo(f"Columns: {list(frame.columns)}")


if __name__ == "__main__":
    app()
