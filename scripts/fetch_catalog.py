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

    uv run scripts/fetch_catalog.py 86.49 8.72 --radius 9 --vmag-max 13

The default filename states the query that produced it -- for that command,
`gaia_086.4900+08.7200_r9.00_g6-13.parquet` -- and an existing, usable file is
left alone rather than re-fetched, so running it again for a field already
pulled costs nothing. The output is what `CatalogSettings.catalog_filename`
expects: parquet, ECSV or CSV by suffix.

See data contract 6 for where camera-level values come from; this is the sky
side of the same problem.
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


def default_name(ra: float, dec: float, radius: float, vmag_min: float, vmag_max: float) -> str:
    """A filename that states the query that produced it.

    ``gaia_086.4900+08.7200_r9.00_g6-13.parquet``. Every argument that changes
    the result is in the name, so two different cones cannot collide and the
    same cone always lands on the same path -- which is what makes the
    existence check below able to skip the query rather than repeat it.

    Dec carries an explicit sign, as it does everywhere else in astronomy, so
    ``+08`` and ``-08`` are never confused.
    """
    return f"gaia_{ra:08.4f}{dec:+08.4f}_r{radius:.2f}_g{vmag_min:g}-{vmag_max:g}.parquet"


def looks_usable(path: Path) -> bool:
    """True if `path` already holds a catalog with the columns the pipeline needs.

    A bare `exists()` would also skip the query for a truncated download or a
    file left over from a schema change, and the failure would then land much
    later, in catalog matching. Reading the columns is cheap -- parquet keeps
    them in the footer -- and turns "the file is there" into "the file is
    usable".
    """
    import pandas

    try:
        if path.suffix == ".parquet":
            columns = set(pandas.read_parquet(path, columns=None).columns)
        else:
            columns = set(pandas.read_csv(path, nrows=1).columns)
    except Exception as error:  # noqa: BLE001 - any unreadable file is "not usable"
        typer.echo(f"{path} exists but could not be read ({error!r}); re-fetching.")
        return False

    missing = set(COLUMN_MAP.values()) - columns
    if missing:
        typer.echo(f"{path} exists but is missing {sorted(missing)}; re-fetching.")
        return False

    return True


@app.command()
def main(
    ra: float = typer.Argument(..., help="Field centre RA, degrees."),
    dec: float = typer.Argument(..., help="Field centre Dec, degrees."),
    radius: float = typer.Option(9.0, help="Cone radius, degrees."),
    vmag_min: float = typer.Option(6.0, help="Brightest Gaia G to include."),
    vmag_max: float = typer.Option(13.0, help="Faintest Gaia G to include."),
    output: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Where to write. Defaults to a name built from the query arguments.",
    ),
    directory: Path = typer.Option(
        Path("."), "-d", "--directory", help="Directory for the default filename."
    ),
    force: bool = typer.Option(False, "--force", help="Re-query even if the file is present."),
    limit: int = typer.Option(-1, help="Row cap; -1 for no cap."),
) -> None:
    """Fetch a Gaia DR3 cone and write it as a pipeline catalog.

    Skips the query when the output already exists and holds the right columns,
    so re-running for a field that has been fetched once costs nothing. Pass
    `--force` to fetch again -- worth doing if the Gaia data release moves, which
    the filename does not encode.
    """
    output = output or directory / default_name(ra, dec, radius, vmag_min, vmag_max)

    if output.exists() and not force and looks_usable(output):
        import pandas

        rows = len(pandas.read_parquet(output) if output.suffix == ".parquet" else [])
        typer.echo(f"{output} already exists ({rows} rows); not querying. Use --force to refetch.")
        raise typer.Exit()

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
