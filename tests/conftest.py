"""Shared fixtures.

The FITS files under `tests/data/` are real POCS output; see that directory's
README for what each one is and which quirks they carry.
"""

from pathlib import Path

import pytest
from astropy.io import fits
from panoptes.utils.images.fits import ImagePathInfo

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def raw_frame() -> Path:
    """A raw POCS frame: no WCS, no `IMAGEW`/`IMAGEH`."""
    return DATA_DIR / "tiny.fits"


@pytest.fixture
def solved_frame() -> Path:
    """The same frame family after plate solving: WCS present."""
    return DATA_DIR / "solved.fits.fz"


@pytest.fixture
def bare_frame() -> Path:
    """`SIMPLE`, `BITPIX`, `NAXIS*` and nothing else."""
    return DATA_DIR / "noheader.fits"


@pytest.fixture
def raw_header(raw_frame) -> fits.Header:
    return fits.getheader(raw_frame)


@pytest.fixture
def solved_header(solved_frame) -> fits.Header:
    return fits.getheader(solved_frame, ext=1)


@pytest.fixture
def bare_header(bare_frame) -> fits.Header:
    return fits.getheader(bare_frame)


@pytest.fixture
def raw_path_info(raw_header) -> ImagePathInfo:
    return ImagePathInfo.from_fits_header(raw_header)
