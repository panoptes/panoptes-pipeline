from pathlib import Path

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
#: required: CLAUDE.md promises the catalog can be rebuilt without touching this
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


def read_catalog(catalog_filename) -> pandas.DataFrame:
    """Read a catalog file, choosing the reader from its suffix.

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
        raise ValueError(
            f"Catalog {path} has a `picid` column that is not integer identifiers: {e}"
        ) from e

    # An identifier, not a quantity.
    catalog_stars["picid"] = picid.astype("category")

    return catalog_stars


def get_stars_from_coords(ra: float, dec: float, radius: float = 8.0, **kwargs) -> pandas.DataFrame:
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
    wcs0: WCS, round_to: int = 0, pad: float = 1.0, pad_size=(20, 10), **kwargs
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
        **kwargs: Optional keywords to pass to :py:func:`get_stars`.

    """
    wcs_footprint = wcs0.calc_footprint()
    print(f"Looking up catalog stars for WCS: {wcs_footprint}")

    ra_max, dec_max = (wcs0.wcs.crval + np.array(pad_size)).round(round_to)
    ra_min, dec_min = (wcs0.wcs.crval - np.array(pad_size)).round(round_to)

    limits = dict(ra_max=ra_max % 360, ra_min=ra_min % 360, dec_max=dec_max, dec_min=dec_min)

    print(f"Searching square shape with {round_to=} and {pad=}: {limits!r}")
    catalog_stars = get_stars(shape=limits, **kwargs)

    return catalog_stars


def get_stars(shape=None, vmag_min=7, vmag_max=14, catalog_filename=None, **kwargs):
    """Look up star information from a local copy of the PANOPTES Input Catalog.

    The PIC is derived from the [TESS Input Catalog](
    https://tess.mit.edu/science/tess-input-catalogue/) v8. It is read from a
    local file -- parquet, ECSV or CSV, see :py:func:`read_catalog` -- and getting
    that file onto disk is a separate fetch step, not something this function
    does. There is no network lookup; see improvement plan 4.3.

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
        **kwargs: Ignored, for call-site compatibility.

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
    **kwargs,
):
    """Match the point source positions to the catalog.

    The catalog is matched to the PANOPTES Input Catalog (PIC), which is derived
    from the [TESS Input Catalog](https://tess.mit.edu/science/tess-input-catalogue/)
    [v8](https://heasarc.gsfc.nasa.gov/docs/tess/tess-input-catalog-version-8-tic-8-is-now-available-at-mast.html).

    The catalog is read from a local file. This function will match the
    `measured_ra` and `measured_dec` columns (as output from `lookup_point_sources`)
    to the `catalog_ra` and `catalog_dec` columns of the catalog. The actual lookup
    is done via :py:func:`get_stars_from_wcs`.

    The columns are added to `point_sources`, which is then returned to the user.

    Columns that are added to `point_sources` include:

        * picid
        * unit_id
        * camera_id
        * time
        * gaia
        * twomass
        * catalog_dec
        * catalog_ra
        * catalog_sep_arcsec
        * catalog_measured_diff_arcsec_dec
        * catalog_measured_diff_arcsec_ra
        * catalog_measured_diff_x
        * catalog_measured_diff_y
        * catalog_vmag
        * catalog_vmag_err
        * catalog_x
        * catalog_y
        * catalog_x_int
        * catalog_y_int

    Note:

        Note all fields are expected to have values. In particular, the `gaia`
        and `twomass` fields are often mutually exclusive.  If `return_unmatched=True`
        (see below) then all values related to matching will be `NA` for all `photutils`
        related columns.

    By default only the sources that are successfully matched by the catalog are returned.
    This behavior can be changed by setting `return_unmatched=True`. This will append
    *all* catalog entries within the Vmag range [vmag_min, vmag_max).

    Warning:

        Using `return_unmatched=True` can return a very large datafraame depending
        on the chosen Vmag range and galactic coordinates. However, it should be
        noted that limiting the Vmag range makes results less accurate.

        The best policy would be to try to minimize calls to this function. The
        resulting dataframe can be saved locally with `point_sources.to_csv(path_name)`.

    If a `max_separation_arcsec` is given then results will be filtered if their
    match with `photutils` was larger than the number given. Typical values would
    be in the range of 20-30 arcsecs, which corresponds to 2-3 pixels.

    Returns:
        `pandas.DataFrame`: A dataframe with the catalog information added to the
        sources.

    Args:
        point_sources (`pandas.DataFrame`): The DataFrame containing point sources
            to be matched. This usually comes from the output of `lookup_point_sources`
            but could be done manually.
        wcs (`astropy.wcs.WCS`, optional): The WCS instance to use for the catalog lookup.
            Either the `wcs` or the `catalog_stars` must be supplied.
        catalog_stars (`pandas.DataFrame`, optional): If provided, the catalog match
            will be performed against this set of stars rather than performing a lookup.
        ra_column (str): The column name to use for the RA coordinates, default `measured_ra`.
        dec_column (str): The column name to use for the Dec coordinates, default `measured_dec`.
        origin (int, optional): The origin for catalog matching, either 0 or 1 (default).
        max_separation_arcsec (float|None, optional): If not None, sources more
            than this many arcsecs from catalog will be filtered.
        return_unmatched (bool, optional): If all results from catalog should be
            returned, not just those with a positive match.
        origin (int): The origin for the WCS. See `all_world2pix`. Default 1.
        **kwargs: Extra options are passed to `get_stars_from_wcs`, which
            passes them to `get_stars`.

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
    matched_sources = point_sources.reset_index(drop=True).join(
        catalog_matches.reset_index(drop=True)
    )

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
