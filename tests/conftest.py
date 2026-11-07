"""Shared fixtures.

The FITS files under `tests/data/` are real POCS output; see that directory's
README for what each one is and which quirks they carry.
"""

import json
import shutil
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


@pytest.fixture
def make_raw_tree(tmp_path, raw_frame):
    """Build a raw tree of real frames with distinct, controllable identities.

    The POCS fixtures all carry `IMAGEID == SEQID`, so a walk over unedited
    copies would see one frame repeated. This rewrites both keywords per frame,
    which is also what makes the layout on disk match what `ImagePathInfo`
    parses out of the header.
    """

    def build(
        count: int = 3,
        unit_id: str = "PAN001",
        camera_id: str = "abc123",
        sequence_time: str = "20220115T082108",
        root_name: str = "raw",
    ) -> Path:
        root = tmp_path / root_name
        data = fits.getdata(raw_frame)
        header = fits.getheader(raw_frame)

        for index in range(count):
            image_time = f"20220115T0822{index:02d}"
            frame_header = header.copy()
            frame_header["SEQID"] = f"{unit_id}_{camera_id}_{sequence_time}"
            frame_header["IMAGEID"] = f"{unit_id}_{camera_id}_{image_time}"

            path = root / unit_id / camera_id / sequence_time / f"{image_time}.fits"
            path.parent.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(data, header=frame_header).writeto(path)

        return root

    return build


@pytest.fixture
def widefield(tmp_path) -> Path:
    """A copy of the wide-field frame, because solving destroys its input.

    `get_solve_field` unpacks a compressed file and does not restore it when
    `replace=False`, so solving the committed fixture in place would delete it
    from the working tree.
    """
    target = tmp_path / "widefield.fits.fz"
    shutil.copy(DATA_DIR / "widefield.fits.fz", target)
    return target


@pytest.fixture
def widefield_registry(tmp_path) -> Path:
    """A camera registry holding a profile for `widefield.fits.fz` (PAN025 / df3495).

    The registry is data outside this repository (data contract 6.2), so a test
    that needs a known camera writes its own. These are that camera's values in
    the operational registry: black and clip by body and histogram, gain the
    24 MP family value.
    """
    root = tmp_path / "registry"
    path = root / "units" / "PAN025" / "cameras" / "df3495.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            dict(
                unit_id="PAN025",
                camera_uid="df3495",
                serial_number="273074013298",
                sensor="6020x4016",
                iso=100,
                black_level=dict(value=2048, method="body"),
                clip=dict(value=13583, method="histogram"),
                gain=dict(value=2.1, uncertainty=0.2, method="family"),
                read_noise=dict(value=7.0, method="sky-ptc"),
            )
        )
    )
    return root


@pytest.fixture
def widefield_catalog() -> Path:
    """The trimmed Gaia catalog covering `widefield.fits.fz`."""
    return DATA_DIR / "widefield_catalog.parquet"
