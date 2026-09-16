"""Calibration, and the parts of the entry points that can run without a solver.

Plate solving hard-requires astrometry.net's `solve-field`, so anything past
the calibration step is skipped where the binary is absent. That is stated
rather than worked around: a frame with no WCS cannot be matched to the
catalog, so there is nothing useful to write, and a flag that produced
unmatched frames would just move the failure downstream.
"""

import os
import shutil

import numpy as np
import pandas
import pytest
from astropy.io import fits

from panoptes.pipeline import processing, products, worklist
from panoptes.pipeline.processing import SolverMissing
from panoptes.pipeline.settings import CameraSettings, ImageSettings, PipelineParams
from panoptes.pipeline.status import ImageStatus
from panoptes.pipeline.utils.images import extract_metadata

#: An end-to-end run needs three things this repository cannot ship: the
#: `solve-field` binary, astrometry.net index files covering a 10-20 degree
#: field, and a local catalog to match against. The `solve-field` binary alone
#: is not enough -- it is installed here and still cannot solve anything,
#: because no index files are present. Point these at a real wide-field frame
#: and catalog to run it.
REAL_FRAME = os.environ.get("PANOPTES_TEST_FRAME")
REAL_CATALOG = os.environ.get("PANOPTES_TEST_CATALOG")

needs_real_data = pytest.mark.skipif(
    not (REAL_FRAME and REAL_CATALOG and shutil.which("solve-field")),
    reason="needs solve-field, index files, PANOPTES_TEST_FRAME and PANOPTES_TEST_CATALOG",
)


@pytest.fixture
def params() -> PipelineParams:
    # The POCS test frames are 200x200, far smaller than a real frame, so the
    # background mesh has to be small enough to fit inside them.
    return PipelineParams.model_validate(
        {"background": {"box_size": (40, 40), "filter_size": (3, 3)}}
    )


# --- calibration ----------------------------------------------------------


def test_calibrating_subtracts_the_background(raw_header, params):
    data = np.random.default_rng(0).normal(1000, 10, (200, 200))

    calibrated = processing.calibrate(data, raw_header, params)

    assert abs(float(np.ma.median(calibrated.reduced))) < abs(float(np.median(data)))


def test_saturation_is_masked_in_the_raw_domain(raw_header, params):
    """`WHTLVLN` is a raw white level, so the comparison happens before bias."""
    white_level = raw_header["WHTLVLN"]
    data = np.full((200, 200), 1000.0)
    data[0, 0] = white_level
    data[0, 1] = white_level - 1

    calibrated = processing.calibrate(data, raw_header, params)

    assert calibrated.mask[0, 0]
    assert not calibrated.mask[0, 1]


def test_the_threshold_comes_from_the_header_not_the_default(raw_header, params):
    """The whole point of resolving saturation: the default would miss these."""
    white_level = raw_header["WHTLVLN"]
    assert white_level < CameraSettings().saturation

    data = np.full((200, 200), 1000.0)
    data[5, 5] = white_level + 10

    calibrated = processing.calibrate(data, raw_header, params)

    assert calibrated.mask[5, 5]
    assert calibrated.calibration["saturation"].source == "WHTLVLN"


def test_a_frame_with_no_saturated_pixels_still_calibrates(raw_header, params):
    """`np.ma` uses a scalar `nomask` here, which used to build a 0-d mask."""
    data = np.full((200, 200), 1000.0)

    calibrated = processing.calibrate(data, raw_header, params)

    assert calibrated.mask.shape == data.shape
    assert not calibrated.mask.any()


def test_the_background_mesh_is_kept_per_colour(raw_header, params):
    """Summing the colours is what loses information; the mesh does not."""
    data = np.random.default_rng(0).normal(1000, 10, (200, 200))

    calibrated = processing.calibrate(data, raw_header, params)

    assert calibrated.background_mesh.shape[0] == 3
    assert calibrated.background_mesh.ndim == 3
    assert calibrated.background.shape == data.shape


def test_the_mesh_is_far_smaller_than_its_interpolation(raw_header, params):
    data = np.random.default_rng(0).normal(1000, 10, (200, 200))

    calibrated = processing.calibrate(data, raw_header, params)

    assert calibrated.background_mesh.size < calibrated.background.size


def test_calibration_does_not_mutate_the_params(raw_header, params):
    """Writing the frame shape back into params would break the fingerprint."""
    before = params.fingerprint
    processing.calibrate(np.full((200, 200), 1000.0), raw_header, params)

    assert params.fingerprint == before


# --- the solver gate ------------------------------------------------------


def test_processing_fails_loudly_without_a_solver(tmp_path, make_raw_tree, params, monkeypatch):
    monkeypatch.setattr(processing.shutil, "which", lambda _: None)
    raw_root = make_raw_tree(count=1)

    with pytest.raises(SolverMissing, match="solve-field"):
        processing.process_frame(
            next(worklist.find_frames(raw_root)), tmp_path / "processed", params
        )


def test_the_solver_check_happens_before_any_pixels_are_read(tmp_path, params, monkeypatch):
    monkeypatch.setattr(processing.shutil, "which", lambda _: None)

    with pytest.raises(SolverMissing):
        processing.process_frame(tmp_path / "does-not-exist.fits", tmp_path / "processed", params)


def test_solving_never_replaces_the_file_it_is_given(monkeypatch, tmp_path):
    """`solve-field` rewrites a file as a single HDU, discarding extensions.

    `panoptes-utils` defaults `replace` to True, so solving a multi-extension
    product in place silently destroys BACKGROUND, RMS and MASK. Confirmed
    against a real frame before this was pinned.
    """
    from panoptes.pipeline.utils import images as image_utils

    captured = {}

    def fake_solve(fname, **kwargs):
        captured.update(kwargs)
        return {"solved_fits_file": fname, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"}

    monkeypatch.setattr(image_utils.fits_utils, "get_solve_field", fake_solve)
    image_utils.plate_solve(
        settings=ImageSettings(params=PipelineParams(), output_dir=tmp_path),
        filename=tmp_path / "image.fits",
    )

    assert captured["replace"] is False


def test_the_solve_timeout_is_the_one_passed(monkeypatch, tmp_path):
    """It was hardcoded to 300, so the argument did nothing."""
    from panoptes.pipeline.utils import images as image_utils

    captured = {}

    def fake_solve(fname, **kwargs):
        captured.update(kwargs)
        return {"solved_fits_file": fname}

    monkeypatch.setattr(image_utils.fits_utils, "get_solve_field", fake_solve)
    image_utils.plate_solve(
        settings=ImageSettings(params=PipelineParams(), output_dir=tmp_path),
        filename=tmp_path / "image.fits",
        timeout=17,
    )

    assert captured["timeout"] == 17


# --- the sequence document ------------------------------------------------


def test_the_sequence_document_is_written_once_per_sequence(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=3)
    processed = tmp_path / "processed"
    frames = worklist.build(raw_root, processed, params)

    written = processing.write_observation(processed, frames, params)

    assert len(written) == 1
    assert list(written)[0] == frames[0].path_info.sequence_id
    assert list(processed.rglob("observation.json")) == list(written.values())


def test_the_sequence_document_sits_above_the_frames(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=2)
    processed = tmp_path / "processed"
    frames = worklist.build(raw_root, processed, params)

    path = next(iter(processing.write_observation(processed, frames, params).values()))

    assert path.parent == products.frame_directory(processed, frames[0].path_info).parent


def test_an_unprocessed_sequence_is_recorded_as_an_error(tmp_path, make_raw_tree, params):
    """No frame produced a document, so the sequence did not happen."""
    raw_root = make_raw_tree(count=2)
    processed = tmp_path / "processed"
    frames = worklist.build(raw_root, processed, params)

    path = next(iter(processing.write_observation(processed, frames, params).values()))
    document = products.read_document(path)

    assert document["status"] == "ERROR"
    assert document["num_frames"] == 2
    assert document["num_processed"] == 0


def test_a_fully_processed_sequence_is_recorded_as_matched(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=2)
    processed = tmp_path / "processed"
    frames = worklist.build(raw_root, processed, params)

    for frame in frames:
        header = fits.getheader(frame.raw_path)
        metadata = products.record_processing(
            extract_metadata(header, frame.path_info), params, ImageStatus.MATCHED
        )
        products.write_frame(processed, frame.path_info, metadata)

    path = next(iter(processing.write_observation(processed, frames, params).values()))
    document = products.read_document(path)

    assert document["status"] == "MATCHED"
    assert document["num_processed"] == 2
    assert document["params_fingerprint"] == params.fingerprint


def test_frames_group_by_their_sequence(tmp_path, make_raw_tree, params):
    first = make_raw_tree(count=2, sequence_time="20220115T082108", root_name="a")
    second = make_raw_tree(count=1, sequence_time="20220115T090000", root_name="b")

    frames = worklist.build(first, tmp_path / "p", params) + worklist.build(
        second, tmp_path / "p", params
    )

    assert len(processing.group_by_sequence(frames)) == 2


def test_sequences_are_discoverable_without_processing(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=3)

    assert processing.find_sequences(raw_root) == ["PAN001_abc123_20220115T082108"]


# --- end to end -----------------------------------------------------------


@needs_real_data
def test_a_frame_processes_end_to_end(tmp_path):
    """The only test that exercises solve, detect and match together."""
    params = PipelineParams.model_validate({"catalog": {"catalog_filename": REAL_CATALOG}})

    written = processing.process_frame(REAL_FRAME, tmp_path / "processed", params)

    assert set(written) == {"metadata", "reduced", "sources"}
    document = products.read_document(written["metadata"])
    assert document["image"]["status"] == "MATCHED"
    assert document["image"]["sources"]["num_detected"] > 0


def test_a_failed_frame_is_recorded_so_the_next_walk_finds_it(
    tmp_path, make_raw_tree, params, monkeypatch
):
    """A frame that cannot be solved still leaves a document saying so.

    The solver is stubbed in both directions -- present, then failing -- so the
    failure happens inside the guarded stage rather than at the gate. Relying
    on a real `solve-field` being installed (or absent) would make the test
    assert something different depending on the machine.
    """
    monkeypatch.setattr(processing.shutil, "which", lambda _: "/usr/bin/solve-field")
    monkeypatch.setattr(
        processing,
        "plate_solve",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no solution")),
    )

    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"

    with pytest.raises(RuntimeError, match="no solution"):
        processing.process_frame(raw_path, processed, params)

    frame = worklist.decide(raw_path, processed, params)
    assert frame.status is ImageStatus.ERROR
    assert frame.reason is worklist.Reason.PRIOR_ERROR


def test_a_rerun_with_nothing_to_do_needs_no_solver(tmp_path, make_raw_tree, params, monkeypatch):
    """Confirming there is nothing to do is the normal case, not the exception."""
    monkeypatch.setattr(processing.shutil, "which", lambda _: None)
    raw_root = make_raw_tree(count=2)
    processed = tmp_path / "processed"

    for frame in worklist.build(raw_root, processed, params):
        metadata = products.record_processing(
            extract_metadata(fits.getheader(frame.raw_path), frame.path_info),
            params,
            ImageStatus.MATCHED,
        )
        products.write_frame(processed, frame.path_info, metadata)

    table = processing.process_observation(raw_root, processed, params)

    assert set(table.outcome) == {"skipped"}


def test_a_rerun_with_work_to_do_still_demands_a_solver(
    tmp_path, make_raw_tree, params, monkeypatch
):
    monkeypatch.setattr(processing.shutil, "which", lambda _: None)
    raw_root = make_raw_tree(count=1)

    with pytest.raises(SolverMissing):
        processing.process_observation(raw_root, tmp_path / "processed", params)


def test_the_document_is_written_after_the_products(tmp_path, raw_path_info, raw_header, params):
    """The document is the completion marker, so it must be committed last.

    A failure part way through the products would otherwise leave a frame
    claiming to be finished, and the next walk would skip it.
    """
    metadata = products.record_processing(
        extract_metadata(raw_header, raw_path_info), params, ImageStatus.MATCHED
    )

    class Exploding(pandas.DataFrame):
        def to_parquet(self, *args, **kwargs):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        products.write_frame(tmp_path, raw_path_info, metadata, sources=Exploding({"picid": [1]}))

    document = products.frame_directory(tmp_path, raw_path_info) / "metadata.json"
    assert not document.exists()
