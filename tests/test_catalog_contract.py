"""A catalog with the four documented columns has to actually work.

AGENTS.md promises the catalog can be rebuilt without touching this package,
and `REQUIRED_CATALOG_COLUMNS` names four. `match_sources` then read three Gaia
columns unconditionally, so a catalog built to the documented contract
validated, matched, and died -- after solving and source detection had already
run. See #202.
"""

import numpy as np
import pandas
import pytest
from astropy.wcs import WCS

from panoptes.pipeline.settings import CatalogSettings, ImageSettings, PipelineParams
from panoptes.pipeline.utils import sources
from panoptes.pipeline.utils.images import match_sources

CENTRE_RA, CENTRE_DEC = 86.5, 8.7
EXCESS_COLUMNS = (
    "catalog_gaia_bg_excess",
    "catalog_gaia_br_excess",
    "catalog_gaia_rg_excess",
)


@pytest.fixture
def wcs() -> WCS:
    solved = WCS(naxis=2)
    solved.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    solved.wcs.crval = [CENTRE_RA, CENTRE_DEC]
    solved.wcs.crpix = [1000, 1000]
    solved.wcs.cdelt = [-0.0025, 0.0025]
    return solved


@pytest.fixture
def catalog(tmp_path):
    """Write a catalog, with or without the optional Gaia photometry."""

    def build(with_gaia: bool, count: int = 200, name: str = "catalog.parquet"):
        rng = np.random.default_rng(0)
        frame = pandas.DataFrame(
            {
                "picid": np.arange(1, count + 1, dtype="int64"),
                "catalog_ra": CENTRE_RA + rng.uniform(-1, 1, count),
                "catalog_dec": CENTRE_DEC + rng.uniform(-1, 1, count),
                "catalog_vmag": rng.uniform(7, 12, count),
            }
        )
        if with_gaia:
            frame["catalog_gaiabp"] = frame.catalog_vmag + 0.4
            frame["catalog_gaiarp"] = frame.catalog_vmag - 0.4
            frame["catalog_gaiamag"] = frame.catalog_vmag

        path = tmp_path / name
        frame.to_parquet(path, index=False)
        return path, frame

    return build


def detected_from(catalog_frame: pandas.DataFrame, wcs: WCS) -> pandas.DataFrame:
    """Detections sitting on the catalog stars, as a solved frame would give."""
    x, y = wcs.world_to_pixel_values(catalog_frame.catalog_ra, catalog_frame.catalog_dec)
    return pandas.DataFrame(
        {
            "photutils_sky_centroid_ra": catalog_frame.catalog_ra,
            "photutils_sky_centroid_dec": catalog_frame.catalog_dec,
            "photutils_x_centroid": x,
            "photutils_y_centroid": y,
            "photutils_fwhm": 2.4,
        }
    )


def settings_for(path, tmp_path) -> ImageSettings:
    return ImageSettings(
        params=PipelineParams(catalog=CatalogSettings(catalog_filename=path)),
        output_dir=tmp_path,
    )


def test_a_four_column_catalog_matches(catalog, wcs, tmp_path):
    """The documented contract, end to end through the function that broke."""
    path, frame = catalog(with_gaia=False)

    matched = match_sources(
        detected_from(frame, wcs),
        wcs,
        settings_for(path, tmp_path),
        image_width=2000,
        image_height=2000,
    )

    assert len(matched) > 0
    assert "picid" in matched.columns


def test_a_four_column_catalog_gets_no_color_excesses(catalog, wcs, tmp_path):
    """Skipped, not faked: absent photometry must not become a zero color."""
    path, frame = catalog(with_gaia=False)

    matched = match_sources(
        detected_from(frame, wcs),
        wcs,
        settings_for(path, tmp_path),
        image_width=2000,
        image_height=2000,
    )

    assert not [c for c in EXCESS_COLUMNS if c in matched.columns]


def test_a_catalog_with_gaia_photometry_gets_the_excesses(catalog, wcs, tmp_path):
    path, frame = catalog(with_gaia=True)

    matched = match_sources(
        detected_from(frame, wcs),
        wcs,
        settings_for(path, tmp_path),
        image_width=2000,
        image_height=2000,
    )

    assert all(c in matched.columns for c in EXCESS_COLUMNS)
    assert matched.catalog_gaia_br_excess.iloc[0] == pytest.approx(0.8)


def test_the_required_columns_are_the_documented_four():
    assert sources.REQUIRED_CATALOG_COLUMNS == (
        "picid",
        "catalog_ra",
        "catalog_dec",
        "catalog_vmag",
    )


def test_the_gaia_columns_are_not_required():
    assert not set(sources.GAIA_CATALOG_COLUMNS) & set(sources.REQUIRED_CATALOG_COLUMNS)


def test_a_catalog_missing_a_required_column_fails_at_read_time(catalog, tmp_path):
    """Loudly, and before any expensive work, unlike the Gaia columns did."""
    path, frame = catalog(with_gaia=False)
    frame.drop(columns=["catalog_vmag"]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="catalog_vmag"):
        sources.read_catalog(path)
