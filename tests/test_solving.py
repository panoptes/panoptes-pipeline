"""Plate solving, run for real against a frame at PANOPTES angular scale.

Everything from the solve call onwards used to be verified only by hand, which
is how three defects reached `main`: `solve-field` destroying every FITS
extension but the first, source detection unreachable behind an undeclared
`scikit-image`, and a masked array handed to `photutils`. Each was found by
running the pipeline, not by the suite.

These need `solve-field` and index files covering a ten-to-twenty degree field.
CI installs both; a developer machine without them skips.
"""

import shutil

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from panoptes.pipeline import processing, products
from panoptes.pipeline.settings import ImageSettings, PipelineParams
from panoptes.pipeline.utils.images import plate_solve

#: The fixture is `.fz`, so `funpack` is as much a requirement as the solver:
#: `get_solve_field` unpacks a compressed input before handing it over, and
#: without it every solve fails with "no WCS header present" while uncompressed
#: frames still work -- which reads as a fixture problem rather than a missing
#: tool. Naming both here turns that into a skip with a reason.
MISSING_TOOLS = [tool for tool in ("solve-field", "funpack") if shutil.which(tool) is None]

needs_solver = pytest.mark.skipif(
    bool(MISSING_TOOLS),
    reason=f"not installed: {', '.join(MISSING_TOOLS)}",
)

#: The field `widefield.fits.fz` was taken of, from its own solved WCS.
EXPECTED_RA, EXPECTED_DEC = 86.497, 8.729


@pytest.fixture
def widefield(tmp_path, request):
    """A copy of the committed frame, because solving destroys its input.

    `get_solve_field` unpacks a compressed file and does not restore it when
    `replace=False`, so solving the committed fixture in place would delete it
    from the working tree.
    """
    source = request.path.parent / "data" / "widefield.fits.fz"
    target = tmp_path / source.name
    shutil.copy(source, target)
    return target


@needs_solver
def test_a_widefield_frame_solves_with_production_options(widefield, tmp_path):
    """The real hints, the real index series, a real frame."""
    settings = ImageSettings(params=PipelineParams(), output_dir=tmp_path)

    wcs = plate_solve(settings=settings, filename=widefield, timeout=180)

    assert wcs.is_celestial
    assert wcs.wcs.crval[0] == pytest.approx(EXPECTED_RA, abs=0.1)
    assert wcs.wcs.crval[1] == pytest.approx(EXPECTED_DEC, abs=0.1)


@needs_solver
def test_solving_a_product_in_place_keeps_its_extensions(widefield, tmp_path):
    """The panoptes-utils#378 defect, against the real solver rather than a mock.

    `solve-field` rewrites a file as a single HDU instead of adding a WCS to it,
    so solving a multi-extension product discards BACKGROUND, RMS and MASK.
    `plate_solve` defaults to the non-destructive mode to prevent that, and this
    checks the guarantee end to end: the product is what gets solved, so a
    regression deletes real extensions rather than a mock's.
    """
    with fits.open(widefield) as hdul:
        pixels = hdul[1].data.astype(np.float32)

    product = tmp_path / "image.fits"
    products.write_image(
        product,
        pixels,
        background=np.ones((3, 4, 4), dtype=np.float32),
        rms=np.ones((3, 4, 4), dtype=np.float32),
        mask=np.zeros(pixels.shape, dtype=bool),
    )
    assert [hdu.name for hdu in fits.open(product)] == [
        "PRIMARY",
        "BACKGROUND",
        "RMS",
        "MASK",
    ]

    wcs = plate_solve(
        settings=ImageSettings(params=PipelineParams(), output_dir=tmp_path),
        filename=product,
        timeout=180,
    )

    assert wcs.is_celestial
    assert set(processing.read_image(product)) == {"reduced", "background", "rms", "mask"}


@needs_solver
def test_the_solver_writes_nothing_beside_the_product(widefield, tmp_path):
    """Solver leftovers must not reach a frame's product directory."""
    products_dir = tmp_path / "products"
    products_dir.mkdir()
    products.write_image(products_dir / "image.fits", np.zeros((32, 32), dtype=np.float32))

    plate_solve(
        settings=ImageSettings(params=PipelineParams(), output_dir=tmp_path),
        filename=widefield,
        timeout=180,
    )

    assert {p.name for p in products_dir.iterdir()} == {"image.fits"}


@needs_solver
def test_the_solved_wcs_covers_the_whole_frame(widefield, tmp_path):
    """A solution that only fits part of the frame would silently lose stars."""
    wcs = plate_solve(
        settings=ImageSettings(params=PipelineParams(), output_dir=tmp_path),
        filename=widefield,
        timeout=180,
    )

    # The fixture is rebinned, so its angular size is the original frame's.
    corners = wcs.pixel_to_world([0, 1504, 0, 1504], [0, 0, 1003, 1003])
    separations = corners.separation(WCS(wcs.to_header()).pixel_to_world(752, 502)).deg

    assert separations.max() == pytest.approx(8.9, abs=0.5)
