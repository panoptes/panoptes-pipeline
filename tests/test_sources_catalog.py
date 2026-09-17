"""Tests for the local catalog lookup that replaced the BigQuery one.

The cloud path used to do the RA/Dec/Vmag filtering in SQL, and the local path
read the whole file unfiltered. These now have to agree, so the filtering
semantics are worth pinning down: a half-open Vmag range, an inclusive
positional box, and a Right Ascension window that may wrap through zero.

The filtering tests run against every accepted format, because the point of
accepting more than one is that the format does not change the answer.
"""

from __future__ import annotations

import pandas
import pytest
from astropy.table import Table

from panoptes.pipeline.utils import sources

CATALOG_COLUMNS = ["picid", "catalog_ra", "catalog_dec", "catalog_vmag"]

ROWS = [
    (1, 10.0, 5.0, 8.0),
    (2, 20.0, 5.0, 10.0),
    (3, 20.0, 40.0, 10.0),  # outside a narrow dec box
    (4, 350.0, 5.0, 10.0),  # only inside a wrapped RA box
    (5, 20.0, 5.0, 20.0),  # too faint
]


def write_catalog(path, rows=ROWS, columns=CATALOG_COLUMNS):
    """Write a minimal PIC-shaped catalog, picking the writer from the suffix."""
    frame = pandas.DataFrame(rows, columns=columns)
    if ".parquet" in path.suffixes or ".pq" in path.suffixes:
        frame.to_parquet(path)
    elif ".ecsv" in path.suffixes:
        Table.from_pandas(frame).write(path, format="ascii.ecsv", overwrite=True)
    elif ".tsv" in path.suffixes:
        frame.to_csv(path, sep="\t", index=False)
    else:
        frame.to_csv(path, index=False)
    return path


@pytest.fixture(
    params=[
        "pic.parquet",
        "pic.pq",
        "pic.ecsv",
        "pic.ecsv.gz",
        "pic.csv",
        "pic.csv.gz",
        "pic.tsv",
    ]
)
def catalog(request, tmp_path):
    """A catalog in each accepted format, so the tests below run against all of them."""
    return write_catalog(tmp_path / request.param)


def test_no_catalog_path_fails_loudly():
    """A fleet-wide default is what AGENTS.md forbids; this must raise."""
    with pytest.raises(ValueError, match="catalog_filename"):
        sources.get_stars(shape=None)


def test_missing_catalog_file_names_the_path(tmp_path):
    missing = tmp_path / "absent.parquet"
    with pytest.raises(FileNotFoundError, match="absent.parquet"):
        sources.get_stars(catalog_filename=missing)


def test_unrecognized_format_is_rejected(tmp_path):
    path = tmp_path / "pic.fits"
    path.write_bytes(b"not a catalog")

    with pytest.raises(ValueError, match="Unrecognized catalog format"):
        sources.get_stars(catalog_filename=path)


def test_missing_columns_are_named(tmp_path):
    path = tmp_path / "wrong.csv"
    pandas.DataFrame({"picid": [1], "ra": [10.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="catalog_ra"):
        sources.get_stars(catalog_filename=path)


def test_picid_is_categorical_across_formats(catalog):
    """It identifies a star rather than measuring anything."""
    result = sources.get_stars(catalog_filename=catalog, vmag_min=0, vmag_max=30)
    assert isinstance(result.picid.dtype, pandas.CategoricalDtype)
    assert set(result.picid) == {1, 2, 3, 4, 5}


def test_picid_categories_are_integers_not_strings(catalog):
    """CSV would otherwise give '1' where parquet gives 1, and neither matches."""
    result = sources.get_stars(catalog_filename=catalog, vmag_min=0, vmag_max=30)
    assert result.picid.cat.categories.dtype == "int64"


def test_filtering_prunes_unused_categories(catalog):
    """A field cut from an all-sky catalog must not carry every id in the sky."""
    shape = dict(ra_min=0.0, ra_max=15.0, dec_min=0.0, dec_max=10.0)
    result = sources.get_stars(shape=shape, catalog_filename=catalog, vmag_min=0, vmag_max=30)
    assert set(result.picid) == {1}
    assert list(result.picid.cat.categories) == [1]


def test_a_blank_picid_fails_rather_than_becoming_a_float(tmp_path):
    path = tmp_path / "gappy.csv"
    rows = [(1, 10.0, 5.0, 8.0), (None, 20.0, 5.0, 10.0)]
    pandas.DataFrame(rows, columns=CATALOG_COLUMNS).to_csv(path, index=False)

    with pytest.raises(ValueError, match="picid"):
        sources.get_stars(catalog_filename=path)


def test_vmag_range_is_half_open(catalog):
    """Documented as [vmag_min, vmag_max): the lower bound is in, the upper out."""
    result = sources.get_stars(catalog_filename=catalog, vmag_min=8, vmag_max=10)
    assert set(result.picid) == {1}


def test_no_shape_filters_on_vmag_only(catalog):
    result = sources.get_stars(catalog_filename=catalog, vmag_min=0, vmag_max=30)
    assert len(result) == 5


def test_shape_filters_on_the_box(catalog):
    shape = dict(ra_min=0.0, ra_max=30.0, dec_min=0.0, dec_max=10.0)
    result = sources.get_stars(shape=shape, catalog_filename=catalog, vmag_min=0, vmag_max=30)
    # 3 is out on dec, 4 is out on ra, 5 is inside the box and inside this
    # deliberately wide Vmag range.
    assert set(result.picid) == {1, 2, 5}


def test_ra_window_wrapping_through_zero(catalog):
    """ra_max < ra_min means the box straddles 360/0 and the test is an OR."""
    shape = dict(ra_min=340.0, ra_max=15.0, dec_min=0.0, dec_max=10.0)
    result = sources.get_stars(shape=shape, catalog_filename=catalog, vmag_min=0, vmag_max=30)
    assert set(result.picid) == {1, 4}


def test_index_is_reset_so_positional_matching_is_safe(catalog):
    """`get_catalog_match` indexes the result with `.iloc`, then joins on index."""
    shape = dict(ra_min=15.0, ra_max=30.0, dec_min=0.0, dec_max=10.0)
    result = sources.get_stars(shape=shape, catalog_filename=catalog, vmag_min=0, vmag_max=30)
    assert list(result.index) == list(range(len(result)))


def test_the_same_catalog_is_parsed_once(tmp_path, monkeypatch):
    """`get_stars` runs per frame, so parsing per call is parsing per frame."""
    path = write_catalog(tmp_path / "pic.parquet")
    sources.read_catalog.cache_clear()

    parses = []
    real_read_parquet = pandas.read_parquet

    def counting_read_parquet(*args, **kwargs):
        parses.append(args[0])
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pandas, "read_parquet", counting_read_parquet)

    first = sources.read_catalog(path)
    second = sources.read_catalog(path)

    assert len(parses) == 1
    pandas.testing.assert_frame_equal(first, second)


def test_a_rewritten_catalog_is_read_again(tmp_path):
    """A refetched cone at the same path is a different catalog, not a cache hit."""
    path = write_catalog(tmp_path / "pic.parquet")
    sources.read_catalog.cache_clear()

    assert len(sources.read_catalog(path)) == len(ROWS)

    write_catalog(path, rows=ROWS[:2])

    assert len(sources.read_catalog(path)) == 2


def test_mutating_the_result_leaves_the_cached_catalog_alone(tmp_path):
    """One caller adding a column must not hand that column to the next."""
    path = write_catalog(tmp_path / "pic.parquet")
    sources.read_catalog.cache_clear()

    first = sources.read_catalog(path)
    first["scratch"] = 1
    first.loc[0, "catalog_vmag"] = -99.0

    second = sources.read_catalog(path)

    assert "scratch" not in second.columns
    assert second.loc[0, "catalog_vmag"] == ROWS[0][3]


def test_one_file_two_spellings_is_one_cache_entry(tmp_path, monkeypatch):
    """A relative and an absolute path name the same catalog; read it once."""
    path = write_catalog(tmp_path / "pic.parquet")
    sources.read_catalog.cache_clear()

    parses = []
    real_read_parquet = pandas.read_parquet
    monkeypatch.setattr(
        pandas,
        "read_parquet",
        lambda *args, **kwargs: (parses.append(args[0]), real_read_parquet(*args, **kwargs))[1],
    )

    monkeypatch.chdir(tmp_path)
    sources.read_catalog(path)
    sources.read_catalog("pic.parquet")

    assert len(parses) == 1
