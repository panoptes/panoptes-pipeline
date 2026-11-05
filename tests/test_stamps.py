"""The observation-level stamp cube, cut from a processed sequence tree.

Everything here runs against a synthetic processed tree: the per-frame
products `process_frame` would leave, with stars at known positions, so each
assertion can be checked against the pixels that were put there.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas
import pytest
from astropy.io import fits

from panoptes.pipeline import products, stamps
from panoptes.pipeline.settings import FileSettings, StampSettings
from panoptes.pipeline.status import ImageStatus
from panoptes.pipeline.utils.observations import make_stamps
from tests.synthetic import gaussian_psf

WIDTH, HEIGHT = 120, 80
FWHM = 2.5
FILES = FileSettings()

#: Stars by `picid`: (x, y) in pixels. Deliberately not on integer or even
#: pixels, so the superpixel snap has something to do.
STARS = {
    1001: (30.3, 20.7),
    1002: (61.8, 41.2),
    1003: (90.1, 60.4),
    1004: (45.5, 55.9),
    #: Close enough to the edge that a stamp around it would leave the frame.
    1005: (2.1, 40.0),
}
INCOMPLETE = 1004  # matched in every frame but one
MASKED = 1003  # carries a saturated (NaN) pixel in one frame
EDGE = 1005


def frame_time(index: int) -> datetime:
    return datetime(2018, 8, 24, 0, 1, 18, tzinfo=UTC) + timedelta(seconds=35 * index)


def write_frame(
    sequence_dir: Path, index: int, *, status=ImageStatus.MATCHED, rng, fingerprint: str = "abc123def456"
) -> Path:
    """One frame's products: reduced image, matched sources and document."""
    time = frame_time(index)
    directory = sequence_dir / time.strftime("%Y%m%dT%H%M%S")
    directory.mkdir(parents=True)

    image = rng.normal(0.0, 1.0, (HEIGHT, WIDTH)).astype(np.float32)
    rows = []
    for picid, (x, y) in STARS.items():
        image += 5000.0 * gaussian_psf((HEIGHT, WIDTH), (y, x), FWHM / 2.355)
        if picid == INCOMPLETE and index == 1:
            continue
        rows.append(dict(picid=picid, catalog_wcs_x=x + 0.01 * index, catalog_wcs_y=y, catalog_vmag=10.0))
    if index == 2:
        x, y = STARS[MASKED]
        image[int(y), int(x)] = np.nan

    fits.PrimaryHDU(image).writeto(directory / FILES.reduced_filename)
    pandas.DataFrame(rows).to_parquet(directory / FILES.sources_filename)
    products.write_document(
        directory / FILES.metadata_filename,
        dict(
            image=dict(
                status=status.name,
                params_fingerprint=fingerprint,
                image_time=time,
                sources=dict(photutils_fwhm_median=FWHM),
                calibration=dict(
                    image_width=dict(value=WIDTH, provenance="header", source="NAXIS1"),
                    image_height=dict(value=HEIGHT, provenance="header", source="NAXIS2"),
                ),
            )
        ),
    )
    return directory


@pytest.fixture
def sequence(tmp_path) -> Path:
    rng = np.random.default_rng(7)
    sequence_dir = tmp_path / "processed" / "PAN012" / "358d0f" / "20180824T035917"
    for index in range(4):
        write_frame(sequence_dir, index, rng=rng)
    write_frame(sequence_dir, 4, status=ImageStatus.ERROR, rng=rng)
    return sequence_dir


@pytest.fixture
def params() -> StampSettings:
    # Every frame is in the fixture, so the cut has to demand all of them.
    return StampSettings(min_frame_fraction=1.0)


def test_stamp_size_is_even_and_in_fwhm_units(params):
    assert stamps.stamp_size(2.5, params) == 10  # 4 x 2.5 = 10, already even
    assert stamps.stamp_size(2.6, params) == 12  # ceil(10.4) = 11 -> rounded up to even
    assert stamps.stamp_size(0.5, params) == params.min_size  # never below the floor
    assert stamps.stamp_size(1.3, StampSettings(min_size=5)) == 6


def test_the_default_fraction_admits_a_star_missed_in_a_few_cloudy_frames():
    """1.0 was measured to admit nothing on real data; 0.9 is the default for that reason."""
    assert StampSettings().min_frame_fraction == 0.9


def test_an_error_frame_is_not_usable(sequence):
    frames = stamps.usable_frames(sequence)
    assert len(frames) == 4
    assert [frame.time for frame in frames] == sorted(frame.time for frame in frames)
    assert all(frame.fwhm == FWHM for frame in frames)


def test_build_writes_a_cube_with_one_time_axis(sequence, params):
    written = stamps.build(sequence, params)

    assert set(written) == {"stamps", "positions"}
    assert written["stamps"].parent == sequence, "the cube belongs beside observation.json"

    cube, positions, metadata = stamps.read(sequence)
    assert cube.index.names == ["picid", "time"]
    assert metadata["num_frames"] == 4
    assert metadata["stamp_width"] == metadata["stamp_height"] == stamps.stamp_size(FWHM, params)
    assert metadata["frames_fingerprint"] == "abc123def456"
    assert metadata["stamps"] == params.model_dump()

    picids = cube.index.get_level_values("picid").unique()
    assert set(picids) == {1001, 1002}
    assert len(cube) == 4 * len(picids), "every surviving source appears in every usable frame"
    assert set(positions.index) == set(picids), "positions describe exactly what is in the cube"


def test_stamps_are_cut_where_the_positions_say(sequence, params):
    stamps.build(sequence, params)
    cube, positions, metadata = stamps.read(sequence)
    frames = stamps.usable_frames(sequence)

    frame = frames[2]
    image = stamps.read_reduced(frame.directory / FILES.reduced_filename)
    row = positions.loc[1002]
    expected = image[
        int(row.stamp_y_min) : int(row.stamp_y_max), int(row.stamp_x_min) : int(row.stamp_x_max)
    ].ravel()

    pixels = cube.loc[(1002, frame.time)].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(pixels, expected)
    assert len(pixels) == metadata["stamp_width"] * metadata["stamp_height"]


def test_every_stamp_shares_one_bayer_phase(sequence, params):
    stamps.build(sequence, params)
    _, positions, _ = stamps.read(sequence)
    assert (positions.stamp_x_min % 2 == 0).all()
    assert (positions.stamp_y_min % 2 == 0).all()


def test_the_star_sits_inside_its_stamp(sequence, params):
    """The snap moves the origin by at most one pixel, so the star stays near the middle."""
    stamps.build(sequence, params)
    _, positions, _ = stamps.read(sequence)
    for picid, row in positions.iterrows():
        x, y = STARS[picid]
        assert row.stamp_x_min < x < row.stamp_x_max
        assert row.stamp_y_min < y < row.stamp_y_max
        assert abs((row.stamp_x_min + row.stamp_x_max) / 2 - x) <= 1.5
        assert abs((row.stamp_y_min + row.stamp_y_max) / 2 - y) <= 1.5


def test_a_source_missing_from_one_frame_gets_no_stamp(sequence, params):
    stamps.build(sequence, params)
    cube, _, _ = stamps.read(sequence)
    assert INCOMPLETE not in cube.index.get_level_values("picid")


def test_a_lower_frame_fraction_admits_the_incomplete_source(sequence):
    params = StampSettings(min_frame_fraction=0.75)
    stamps.build(sequence, params)
    cube, _, _ = stamps.read(sequence)
    assert INCOMPLETE in cube.index.get_level_values("picid")
    # Its stamp is cut in all four frames, including the one where matching
    # missed it: the position comes from the sequence, not the frame, and the
    # pixels there are real. What the cube does not carry is a per-frame
    # record of which detections failed -- that stays in the sources tables.
    assert len(cube.loc[INCOMPLETE]) == 4


def test_a_masked_source_is_dropped_whole_and_counted(sequence, params):
    stamps.build(sequence, params)
    cube, positions, metadata = stamps.read(sequence)
    assert MASKED not in cube.index.get_level_values("picid")
    assert MASKED not in positions.index
    assert metadata["num_masked_dropped"] == 1
    assert not cube.isna().any().any(), "no NaN survives into the cube"


def test_an_edge_source_is_dropped_before_extraction(sequence, params):
    stamps.build(sequence, params)
    cube, positions, _ = stamps.read(sequence)
    assert EDGE not in positions.index
    assert EDGE not in cube.index.get_level_values("picid")


def test_frames_from_two_parameter_sets_do_not_mix(sequence, params):
    """A cube over frames reduced under different params is a product of neither."""
    write_frame(sequence, 5, rng=np.random.default_rng(3), fingerprint="fedcba987654")
    with pytest.raises(stamps.MixedFingerprints, match="2 fingerprints"):
        stamps.build(sequence, params)


def test_no_usable_frame_is_an_error(tmp_path, params):
    sequence_dir = tmp_path / "PAN012" / "358d0f" / "20180824T035917"
    write_frame(sequence_dir, 0, status=ImageStatus.ERROR, rng=np.random.default_rng(1))
    with pytest.raises(stamps.NoUsableFrames):
        stamps.build(sequence_dir, params)


def test_existing_products_are_not_overwritten_unless_forced(sequence, params):
    stamps.build(sequence, params)
    with pytest.raises(FileExistsError):
        stamps.build(sequence, params, force_new=False)
    stamps.build(sequence, params, force_new=True)


def test_make_stamps_gathers_exactly_the_slices():
    data = np.arange(20 * 30, dtype=float).reshape(20, 30)
    positions = pandas.DataFrame(
        dict(stamp_y_min=[2, 10], stamp_y_max=[6, 14], stamp_x_min=[4, 20], stamp_x_max=[8, 24]),
        index=pandas.Index([7, 3], name="picid"),
    )
    cut = make_stamps(positions, data)
    assert list(cut.index) == [3, 7], "sorted on picid"
    np.testing.assert_array_equal(cut.loc[7].to_numpy(), data[2:6, 4:8].ravel())
    np.testing.assert_array_equal(cut.loc[3].to_numpy(), data[10:14, 20:24].ravel())
    assert list(cut.columns) == [f"pixel_{i:03d}" for i in range(16)]


def test_make_stamps_drops_a_slice_that_leaves_the_frame():
    data = np.zeros((20, 30))
    positions = pandas.DataFrame(
        dict(stamp_y_min=[2, 18], stamp_y_max=[6, 22], stamp_x_min=[4, 4], stamp_x_max=[8, 8]),
        index=pandas.Index([1, 2], name="picid"),
    )
    cut = make_stamps(positions, data)
    assert list(cut.index) == [1]


def test_the_benchmark_loader_reads_the_sequence_directory(sequence, params):
    """`scripts/benchmark_lightcurve.py` is the consumer #94 names; it has to read this."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        from benchmark_lightcurve import load_observation
    finally:
        sys.path.pop(0)

    stamps.build(sequence, params)
    cube, picids, frames, stamp_shape = load_observation(sequence)

    size = stamps.stamp_size(FWHM, params)
    assert stamp_shape == (size, size)
    assert cube.shape == (2, 4, size * size)
    assert list(picids) == [1001, 1002]
    assert len(frames) == 4
