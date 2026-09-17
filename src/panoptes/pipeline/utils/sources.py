from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS

#: Columns the catalog must carry for matching to work. The file is expected to
#: use the mapped PIC names rather than the raw upstream ones.
#: What the pipeline cannot run without. `picid` is exactly the Gaia DR3
#: `source_id` -- an alias and nothing more (#203) -- so a catalog is a Gaia cone
#: search with its columns renamed, and `scripts/fetch_catalog.py` builds one
#: with no crossmatch step.
REQUIRED_CATALOG_COLUMNS = ("picid", "catalog_ra", "catalog_dec", "catalog_vmag")

#: Optional enrichment. Present, `match_sources` derives three color-excess
#: columns for reference selection; absent, it skips them. Deliberately *not*
#: required: AGENTS.md promises the catalog can be rebuilt without touching this
#: package, and a four-column catalog has to actually work for that to be true.
GAIA_CATALOG_COLUMNS = ("catalog_gaiabp", "catalog_gaiarp", "catalog_gaiamag")

#: Catalog file formats, chosen by suffix, each optionally compressed
#: (`.ecsv.gz`, `.csv.bz2`).
#:
#: Parquet is the better default for an all-sky catalog: markedly smaller, and it
#: round-trips dtypes exactly. ECSV is the better default for anything meant to
#: be read or hand-edited -- it is plain text with a YAML header carrying the
#: column types, so it does not have CSV's habit of quietly turning an identifier
#: column into floats. Plain CSV and TSV are accepted so an existing catalog
#: needs no conversion step, but they carry no type information at all; see
#: :py:func:`read_catalog`.
PARQUET_SUFFIXES = (".parquet", ".pq")
ECSV_SUFFIXES = (".ecsv",)
TEXT_SUFFIXES = (".csv", ".tsv")

#: How many catalogs :py:func:`read_catalog` keeps parsed in memory.
#:
#: The cost this avoids is real: nothing calls the reader once. `get_stars` runs
#: per frame, so an all-sky parquet was re-read and re-typed for every frame in
#: a sequence. Two entries, not more, because a cached catalog is a large
#: DataFrame -- one working catalog plus its predecessor, so switching between
#: two fields does not thrash, and a long-lived process cannot accumulate every
#: file it has ever been handed.
CATALOG_CACHE_SIZE = 2


@lru_cache(maxsize=CATALOG_CACHE_SIZE)
def _parse_catalog(path: Path, mtime_ns: int, size: int) -> pandas.DataFrame:
    """Parse and validate one catalog file, memoized on its identity.

    `mtime_ns` and `size` are not used; they are in the signature because they
    are part of the cache key. A catalog rewritten in place -- refetched with
    `--force`, or replaced by a longer cone -- is a different file at the same
    path, and keying on the path alone would serve the old rows for the rest of
    the process. Neither stamp alone is enough: a filesystem with coarse mtime
    can miss a fast rewrite, and a same-length rewrite leaves the size equal.

    `lru_cache` does not memoize exceptions, so an unreadable or malformed
    catalog raises on every call rather than once.
    """
    suffixes = [s.lower() for s in path.suffixes]

    if any(s in suffixes for s in PARQUET_SUFFIXES):
        catalog_stars = pandas.read_parquet(path)
    elif any(s in suffixes for s in ECSV_SUFFIXES):
        catalog_stars = Table.read(path, format="ascii.ecsv").to_pandas()
    elif any(s in suffixes for s in TEXT_SUFFIXES):
        catalog_stars = pandas.read_csv(path, sep="\t" if ".tsv" in suffixes else ",")
    else:
        raise ValueError(
            f"Unrecognized catalog format for {path}. Expected one of "
            f"{list(PARQUET_SUFFIXES + ECSV_SUFFIXES + TEXT_SUFFIXES)}, "
            f"optionally compressed."
        )

    missing = [c for c in REQUIRED_CATALOG_COLUMNS if c not in catalog_stars.columns]
    if missing:
        raise ValueError(
            f"Catalog {path} is missing required column(s): {missing}. "
            f"Expected the mapped PIC names: {list(REQUIRED_CATALOG_COLUMNS)}."
        )

    # Plain CSV has no dtypes, so a single blank turns `picid` into floats that
    # then compare and join as `1234.0` against integer ids everywhere else.
    # Check here rather than match nothing later. Parquet and ECSV both carry the
    # type, so for those this only confirms what the file already declares.
    try:
        picid = catalog_stars["picid"].astype("int64")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Catalog {path} has a `picid` column that is not integer identifiers: {e}") from e

    # An identifier, not a quantity.
    catalog_stars["picid"] = picid.astype("category")

    return catalog_stars


def read_catalog(catalog_filename) -> pandas.DataFrame:
    """Read a catalog file, choosing the reader from its suffix.

    The parsed catalog is cached (:py:data:`CATALOG_CACHE_SIZE`), keyed on the
    file's path, modification time and size, so re-reading the same unchanged
    file is free and a rewritten one is picked up. The cache lives in the
    calling process: a pool of frame workers reads the catalog once each, not
    once per frame. Call ``read_catalog.cache_clear()`` to drop it.

    What comes back is a shallow copy, so a caller that adds or replaces a
    column cannot corrupt the cached catalog for everything after it. Under
    copy-on-write that costs nothing until something is actually written.

    `picid` comes back as a `category`. It identifies a star rather than
    measuring anything, and nothing downstream should be doing arithmetic on it.
    One thing to know when consuming it: `groupby` over a categorical iterates
    every category by default, so pass `observed=True` when grouping on it.

    Args:
        catalog_filename (str|Path): Path to a `.parquet`/`.pq`, `.ecsv` or
            `.csv`/`.tsv` file, optionally compressed (`.ecsv.gz`).

    Returns:
        `pandas.DataFrame`: The catalog, with `picid` as a categorical.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the suffix is not a recognized format, a required column
            is missing, or `picid` cannot be read as an integer identifier.

    """
    path = Path(catalog_filename)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file does not exist: {path}")

    # `resolve` so two spellings of one file -- relative and absolute, or
    # through a symlink -- are one cache entry rather than two copies.
    path = path.resolve()
    stat = path.stat()

    return _parse_catalog(path, stat.st_mtime_ns, stat.st_size).copy(deep=False)


#: The cache is an implementation detail of `read_catalog`, but clearing it is
#: not: a test, or a caller that rewrites a catalog through a path the stamps
#: cannot distinguish, needs a way to start over.
read_catalog.cache_clear = _parse_catalog.cache_clear


def get_stars_from_coords(ra: float, dec: float, radius: float = 8.0, **kwargs: Any) -> pandas.DataFrame:
    limits = dict(
        ra_max=ra + radius,
        ra_min=ra - radius,
        dec_max=dec + radius,
        dec_min=dec - radius,
    )

    print(f"Using {limits=} for get_stars")
    catalog_stars = get_stars(shape=limits, **kwargs)

    return catalog_stars


def get_stars_from_wcs(
    wcs0: WCS, round_to: int = 0, pad: float = 1.0, pad_size=(20, 10), **kwargs: Any
) -> pandas.DataFrame:
    """Lookup star information from WCS footprint.

    Generates the correct layout for an SQL `POLYGON` that can be passed to
    :py:func:`get_stars`.

    Args:
        wcs0 (astropy.wcs.WCS): A valid (i.e. `wcs.is_celestial`) World Coordinate System object.
        round_to (int): Round the limits to this decimal place, default 0. Keeps the
            requested bounds stable between frames of the same sequence.
        pad (float): The amount of padding in degrees to add to each of the RA and Dec
            limits, default 0.5 [degrees].
        **kwargs (Any): Optional keywords to pass to :py:func:`get_stars`.

    """
    wcs_footprint = wcs0.calc_footprint()
    print(f"Looking up catalog stars for WCS: {wcs_footprint}")

    ra_max, dec_max = (wcs0.wcs.crval + np.array(pad_size)).round(round_to)
    ra_min, dec_min = (wcs0.wcs.crval - np.array(pad_size)).round(round_to)

    limits = dict(ra_max=ra_max % 360, ra_min=ra_min % 360, dec_max=dec_max, dec_min=dec_min)

    print(f"Searching square shape with {round_to=} and {pad=}: {limits!r}")
    catalog_stars = get_stars(shape=limits, **kwargs)

    return catalog_stars


def get_stars(shape=None, vmag_min=7, vmag_max=14, catalog_filename=None, **kwargs: Any) -> pandas.DataFrame:
    """Look up star information from a local copy of the PANOPTES Input Catalog.

    `picid` is exactly the Gaia DR3 `source_id`, so the catalog is a Gaia cone
    search with its columns renamed and there is no crossmatch step -- see
    `scripts/fetch_catalog.py`, which builds one. It is read from a local file --
    parquet, ECSV or CSV, see :py:func:`read_catalog` -- and getting that file
    onto disk is a separate fetch step, not something this function does. There
    is no network lookup; see improvement plan 4.3.

    The file is expected to carry the mapped column names
    (:py:data:`REQUIRED_CATALOG_COLUMNS`) rather than the raw upstream ones.

    Note:

        In the upstream catalog the GAIA `bp` and `rp` magnitude and error
        columns are switched. Whatever produces the catalog is responsible for
        correcting that; this function does not second-guess the file it is
        given.

    Args:
        shape (dict|None): A dictionary containing the keys `ra_min`, `ra_max`,
            `dec_min`, `dec_max`, in degrees. If None, no positional filtering.
        vmag_min (float, optional): Minimum Vmag to include, inclusive.
        vmag_max (float, optional): Maximum Vmag to include, exclusive.
        catalog_filename (str|Path): Path to the catalog file: parquet, ECSV
            or CSV. Required; there is no default.
        **kwargs (Any): Ignored, for call-site compatibility.

    Returns:
        `pandas.DataFrame`: The catalog entries inside the requested bounds.

    Raises:
        ValueError: If no catalog path is given, or the file is missing columns.
        FileNotFoundError: If the catalog path does not exist.

    """
    if catalog_filename is None:
        raise ValueError(
            "A local catalog is required. Pass catalog_filename, or set "
            "`params.catalog.catalog_filename` in the pipeline settings. "
            "There is no network catalog lookup."
        )

    catalog_stars = read_catalog(catalog_filename)

    # Vmag range is [vmag_min, vmag_max), as documented.
    selected = catalog_stars.catalog_vmag.between(vmag_min, vmag_max, inclusive="left")

    if shape is not None:
        selected &= catalog_stars.catalog_dec.between(shape["dec_min"], shape["dec_max"])

        # Right Ascension wraps from 360 to 0, so the box can straddle the origin.
        ra = catalog_stars.catalog_ra
        if shape["ra_max"] < shape["ra_min"]:
            selected &= (ra >= shape["ra_min"]) | (ra <= shape["ra_max"])
        else:
            selected &= (ra >= shape["ra_min"]) & (ra <= shape["ra_max"])

    results = catalog_stars[selected].reset_index(drop=True)

    # Filtering a categorical keeps every category, so a field cut from an
    # all-sky catalog would otherwise carry millions of ids that are not in it.
    results["picid"] = results["picid"].cat.remove_unused_categories()

    print(f"Found {len(results)} in Vmag=[{vmag_min}, {vmag_max}) and bounds=[{shape}]")

    return results


def get_catalog_match(
    point_sources,
    wcs=None,
    catalog_stars=None,
    max_separation_arcsec=None,
    ra_column="measured_ra",
    dec_column="measured_dec",
    **kwargs: Any,
) -> pandas.DataFrame:
    """Match the point source positions to the catalog.

    `picid` in the catalog is exactly the Gaia DR3 `source_id`, so the catalog is
    a Gaia cone search with its columns renamed and there is no crossmatch step
    -- see `scripts/fetch_catalog.py`, which builds one.

    The catalog is read from a local file. This function matches the `ra_column`
    and `dec_column` positions to the `catalog_ra` and `catalog_dec` columns of
    the catalog. When `catalog_stars` is not supplied the lookup is done via
    :py:func:`get_stars_from_wcs`.

    The matched catalog row is joined onto each source and the result returned.
    The columns added are whatever the catalog file carries
    (:py:data:`REQUIRED_CATALOG_COLUMNS`, plus :py:data:`GAIA_CATALOG_COLUMNS`
    when the file has them), together with:

        * catalog_sep -- separation from the matched catalog position, in arcsec
        * catalog_wcs_x, catalog_wcs_y -- the catalog position in pixels
        * catalog_wcs_x_int, catalog_wcs_y_int -- the same, truncated to int

    Note:

        Every source is matched to its *nearest* catalog star, so a detection
        with no real counterpart still gets a row -- `max_separation_arcsec` is
        what removes those. Catalog entries with no detection are not returned,
        so the result has at most one row per source in `point_sources`.

    If a `max_separation_arcsec` is given then results will be filtered if their
    separation from the catalog was larger than the number given. Typical values
    would be in the range of 20-30 arcsecs, which corresponds to 2-3 pixels.

    Args:
        point_sources (`pandas.DataFrame`): The DataFrame containing point sources
            to be matched. This usually comes from the output of
            :py:func:`.images.detect_sources` but could be done manually.
        wcs (`astropy.wcs.WCS`): Required, despite the `None` default. Supplying
            `catalog_stars` skips the lookup but not the pixel positions, which
            :py:func:`get_xy_positions` computes from this WCS for every call,
            so omitting it raises `AttributeError` after the matching work is done.
        catalog_stars (`pandas.DataFrame`, optional): If provided, the catalog match
            will be performed against this set of stars rather than performing a lookup.
        max_separation_arcsec (float|None, optional): If not None, sources more
            than this many arcsecs from catalog will be filtered.
        ra_column (str): The column name to use for the RA coordinates, default `measured_ra`.
        dec_column (str): The column name to use for the Dec coordinates, default
            `measured_dec`. Both defaults are stale -- :py:func:`.images.detect_sources`
            names its columns `photutils_sky_centroid_ra` and `..._dec`, which is
            what :py:func:`.images.match_sources` passes.
        **kwargs (Any): Extra options are passed to `get_stars_from_wcs`, which
            passes them to `get_stars`. Only used when `catalog_stars` is None.

    Returns:
        `pandas.DataFrame`: A dataframe with the catalog information added to
            the sources.

    """
    assert point_sources is not None

    if catalog_stars is None:
        print(f"Looking up stars for wcs={wcs.wcs.crval}")
        # Lookup stars in catalog
        catalog_stars = get_stars_from_wcs(wcs, **kwargs)

    if catalog_stars is None:
        print("No catalog matches, returning table without ids")
        return point_sources

    # Get coords for catalog stars
    catalog_coords = SkyCoord(
        ra=catalog_stars["catalog_ra"].values * u.deg,
        dec=catalog_stars["catalog_dec"].values * u.deg,
        frame="icrs",
    )

    # Get coords from detected point sources
    stars_coords = SkyCoord(
        ra=point_sources[ra_column].values * u.deg,
        dec=point_sources[dec_column].values * u.deg,
        frame="icrs",
    )

    # Do catalog matching
    print(f"Matching {len(catalog_coords)} catalog stars to {len(stars_coords)} detected stars")
    idx, d2d, d3d = stars_coords.match_to_catalog_sky(catalog_coords)
    print(f"Got {len(idx)} matched sources (includes duplicates) for wcs={wcs.wcs.crval}")

    catalog_matches = catalog_stars.iloc[idx].copy()
    catalog_matches["catalog_sep"] = d2d.to_value(u.arcsec)

    # Get the XY positions
    catalog_matches = get_xy_positions(wcs, catalog_matches)

    # Add the matches and their separation.
    matched_sources = point_sources.reset_index(drop=True).join(catalog_matches.reset_index(drop=True))

    # All point sources so far are matched.
    # matched_sources['status'] = 'matched'
    # matched_sources.status = matched_sources.status.astype('category')

    # Sources that didn't match.
    #     if return_unmatched:
    #         print(f'Adding unmatched sources to table for wcs={wcs.wcs.crval!r}')
    #         unmatched = catalog_stars.iloc[catalog_stars.index.difference(idx)].copy()

    #         unmatched['status'] = 'unmatched'
    #         point_sources = point_sources.append(unmatched)

    # Reorder columns so id cols are first then alpha.
    # new_column_order = sorted(list(matched_sources.columns))
    # id_cols = ['picid', 'unit_id', 'camera_id', 'time', 'gaia', 'twomass', 'status']
    # for i, col in enumerate(id_cols):
    #     new_column_order.remove(col)
    #     new_column_order.insert(i, col)
    # matched_sources = matched_sources.reindex(columns=new_column_order)

    print(f"Point sources: {len(matched_sources)} for wcs={wcs.wcs.crval!r}")

    # Remove catalog matches that are too far away.
    if max_separation_arcsec is not None:
        print(f"Removing matches > {max_separation_arcsec} arcsec from catalog.")
        matched_sources = matched_sources.query("catalog_sep <= @max_separation_arcsec")

    print(f"Returning matched sources: {len(matched_sources)} for wcs={wcs.wcs.crval!r}")
    return matched_sources


def get_xy_positions(
    wcs_input,
    catalog_df,
    ra_column="catalog_ra",
    dec_column="catalog_dec",
    origin=1,
    copy_catalog=True,
):
    if copy_catalog:
        catalog_df = catalog_df.copy()

    coords = catalog_df[[ra_column, dec_column]]

    # Get the XY positions
    catalog_xy = wcs_input.all_world2pix(coords, origin, ra_dec_order=True)
    catalog_df["catalog_wcs_x"] = catalog_xy.T[0]
    catalog_df["catalog_wcs_y"] = catalog_xy.T[1]
    catalog_df["catalog_wcs_x_int"] = catalog_df.catalog_wcs_x.astype(int)
    catalog_df["catalog_wcs_y_int"] = catalog_df.catalog_wcs_y.astype(int)

    return catalog_df
