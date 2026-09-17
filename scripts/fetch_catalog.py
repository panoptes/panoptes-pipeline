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

import json
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


#: Suffix sets `sources.read_catalog` accepts. Mirrored rather than imported:
#: this script runs in its own environment and must not depend on the package.
PARQUET_SUFFIXES = (".parquet", ".pq")
ECSV_SUFFIXES = (".ecsv",)

#: Where the exact query is stored inside a parquet catalog.
QUERY_METADATA_KEY = b"panoptes_query"


def read_columns(path: Path) -> set[str]:
    """Column names of an existing catalog, by suffix, as `read_catalog` reads them."""
    import pandas

    suffixes = [s.lower() for s in path.suffixes]
    if any(s in suffixes for s in PARQUET_SUFFIXES):
        import pyarrow.parquet as pq

        return set(pq.read_schema(path).names)
    if any(s in suffixes for s in ECSV_SUFFIXES):
        from astropy.table import Table

        return set(Table.read(path, format="ascii.ecsv").colnames)
    return set(pandas.read_csv(path, nrows=1, sep=None, engine="python").columns)


def stored_query(path: Path) -> dict | None:
    """The query a parquet catalog records having been built from, if any."""
    suffixes = [s.lower() for s in path.suffixes]
    if not any(s in suffixes for s in PARQUET_SUFFIXES):
        return None

    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(QUERY_METADATA_KEY)
    return json.loads(raw) if raw else None


def looks_usable(path: Path, query: dict) -> bool:
    """True if `path` already holds the catalog `query` would produce.

    Two checks, because "the file exists" is not the question.

    The columns have to be the ones the pipeline needs. A bare `exists()` would
    also skip the query for a truncated download or a file left over from a
    schema change, and that failure would land much later, in catalog matching.

    The recorded query has to match. The filename rounds its arguments so it
    stays readable, which means two *slightly* different cones can land on the
    same name -- centres a few milliarcseconds apart, or radii differing in the
    third decimal. Those produce catalogs that are equivalent in practice, but
    "in practice" is not something to rely on silently, so the exact arguments
    are written into the parquet metadata and compared here. A catalog written
    before this existed, or in a format with nowhere to put it, falls back to
    the column check with a warning.
    """
    try:
        columns = read_columns(path)
    except Exception as error:  # noqa: BLE001 - any unreadable file is "not usable"
        typer.echo(f"{path} exists but could not be read ({error!r}); re-fetching.")
        return False

    missing = set(COLUMN_MAP.values()) - columns
    if missing:
        typer.echo(f"{path} exists but is missing {sorted(missing)}; re-fetching.")
        return False

    recorded = stored_query(path)
    if recorded is None:
        typer.echo(f"{path} records no query; assuming it matches. Use --force to be sure.")
        return True
    if recorded != query:
        typer.echo(f"{path} was built from a different query ({recorded}); re-fetching.")
        return False

    return True


@app.command()
def main(
    ra: float = typer.Argument(..., help="Field centre RA, degrees."),
    dec: float = typer.Argument(..., help="Field centre Dec, degrees."),
    radius: float = typer.Option(10.0, help="Cone radius, degrees."),
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

    The default radius allows for pointing error. A PANOPTES frame is about
    14.9 x 9.9 degrees, so its corners sit 8.92 degrees from the field centre and
    a 9 degree cone covers them with 5 arcminutes to spare -- but only if it is
    centered on the *solved* position, which is not known until after solving. On
    the frame this was measured against, the mount pointing was 0.379 degrees
    away, and a 9 degree cone centered there would have needed 9.235 to reach the
    far corner. 10 degrees leaves roughly a degree of slack for about 23% more
    rows, which is cheap insurance against silently clipping a corner.
    """
    output = output or directory / default_name(ra, dec, radius, vmag_min, vmag_max)
    query_key = dict(ra=ra, dec=dec, radius=radius, vmag_min=vmag_min, vmag_max=vmag_max, release="gaiadr3")

    if output.exists() and not force and looks_usable(output, query_key):
        typer.echo(f"{output} already exists and matches; not querying. Use --force to refetch.")
        raise typer.Exit()

    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = limit
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    # A cone big enough to cover the frame's corners, filtered on G so the
    # download is the stars the pipeline would actually extract. Sources with no
    # BP or RP are dropped: the color-excess columns would be null and
    # reference selection cannot use them.
    #
    # The magnitude bound is half-open, `>= min` and `< max`, because that is
    # what `get_stars` applies (`inclusive="left"`). ADQL `BETWEEN` includes
    # both ends, which would fetch sources at exactly `vmag_max` that the
    # pipeline then discards.
    query = f"""
        SELECT source_id, ra, dec,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(POINT({ra}, {dec}), CIRCLE(ra, dec, {radius}))
          AND phot_g_mean_mag >= {vmag_min}
          AND phot_g_mean_mag < {vmag_max}
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
    # things -- brightness selection and color excess -- and a later change to
    # either should not silently move the other.
    frame["catalog_gaiamag"] = frame["catalog_vmag"]

    frame["picid"] = frame["picid"].astype("int64")
    frame = frame[list(dict.fromkeys(COLUMN_MAP.values()))]

    output.parent.mkdir(parents=True, exist_ok=True)
    suffixes = [s.lower() for s in output.suffixes]
    if any(s in suffixes for s in PARQUET_SUFFIXES):
        # Write through pyarrow so the exact query travels with the catalog.
        # The filename is a readable cache key; this is the exact one.
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame, preserve_index=False)
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), QUERY_METADATA_KEY: json.dumps(query_key).encode()}
        )
        pq.write_table(table, output)
    elif any(s in suffixes for s in ECSV_SUFFIXES):
        from astropy.table import Table

        written = Table.from_pandas(frame)
        written.meta["panoptes_query"] = query_key
        written.write(output, format="ascii.ecsv", overwrite=True)
    else:
        frame.to_csv(output, index=False)

    size = output.stat().st_size / 1e6
    typer.echo(f"Wrote {len(frame)} rows to {output} ({size:.1f} MB)")
    typer.echo(f"Columns: {list(frame.columns)}")


if __name__ == "__main__":
    app()
