"""Tests for the local catalog lookup that replaced the BigQuery one.

The cloud path used to do the RA/Dec/Vmag filtering in SQL, and the local path
read the whole parquet unfiltered. These now have to agree, so the filtering
semantics are worth pinning down: a half-open Vmag range, an inclusive
positional box, and a Right Ascension window that may wrap through zero.
"""

from __future__ import annotations

import pandas
import pytest

from panoptes.pipeline.utils import sources


def write_catalog(path, rows):
    """Write a minimal PIC-shaped parquet and return its path."""
    frame = pandas.DataFrame(rows, columns=["picid", "catalog_ra", "catalog_dec", "catalog_vmag"])
    frame.to_parquet(path)
    return path


@pytest.fixture
def catalog(tmp_path):
    return write_catalog(
        tmp_path / "pic.parquet",
        [
            (1, 10.0, 5.0, 8.0),
            (2, 20.0, 5.0, 10.0),
            (3, 20.0, 40.0, 10.0),  # outside a narrow dec box
            (4, 350.0, 5.0, 10.0),  # only inside a wrapped RA box
            (5, 20.0, 5.0, 20.0),  # too faint
        ],
    )


def test_no_catalog_path_fails_loudly(tmp_path):
    """A fleet-wide default is what CLAUDE.md forbids; this must raise."""
    with pytest.raises(ValueError, match="catalog_filename"):
        sources.get_stars(shape=None)


def test_missing_catalog_file_names_the_path(tmp_path):
    missing = tmp_path / "absent.parquet"
    with pytest.raises(FileNotFoundError, match="absent.parquet"):
        sources.get_stars(catalog_filename=missing)


def test_missing_columns_are_named(tmp_path):
    path = tmp_path / "wrong.parquet"
    pandas.DataFrame({"picid": [1], "ra": [10.0]}).to_parquet(path)

    with pytest.raises(ValueError, match="catalog_ra"):
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
