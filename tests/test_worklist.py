"""The work list must say what will be processed and why, before anything runs.

The defect being closed is specific: the old flow stored `params` in every
document and never compared them, so a settings change left stale products in
place with nothing to signal it. The fingerprint is the cache key, and these
check that it is actually used as one. See data contract 5.1 and 5.2.
"""

import pandas
import pytest
from astropy.io import fits
from panoptes.utils.images.fits import ImagePathInfo

from panoptes.pipeline import products, worklist
from panoptes.pipeline.settings import CameraSettings, FileSettings, PipelineParams
from panoptes.pipeline.status import ImageStatus
from panoptes.pipeline.utils.images import extract_metadata
from panoptes.pipeline.worklist import Reason


@pytest.fixture
def params() -> PipelineParams:
    return PipelineParams()


def process(root, raw_path, params, status=ImageStatus.MATCHED):
    """Write the document a processed frame would leave behind."""
    header = fits.getheader(raw_path)
    path_info = ImagePathInfo.from_fits_header(header)

    metadata = products.record_processing(extract_metadata(header, path_info), params, status)
    return products.write_frame(root, path_info, metadata)


# --- the fingerprint ------------------------------------------------------


def test_the_fingerprint_is_stable_across_identical_settings():
    assert PipelineParams().fingerprint == PipelineParams().fingerprint


def test_the_fingerprint_changes_when_a_parameter_changes():
    changed = PipelineParams(camera=CameraSettings(saturation=11765))

    assert changed.fingerprint != PipelineParams().fingerprint


def test_the_fingerprint_is_short_enough_to_read():
    assert len(PipelineParams().fingerprint) == 12


# --- the decision ---------------------------------------------------------


def test_an_unprocessed_frame_is_missing(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))

    frame = worklist.decide(raw_path, tmp_path / "processed", params)

    assert frame.reason is Reason.MISSING
    assert frame.needs_processing


def test_a_processed_frame_at_the_same_settings_is_skipped(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"
    process(processed, raw_path, params)

    frame = worklist.decide(raw_path, processed, params)

    assert frame.reason is Reason.UP_TO_DATE
    assert not frame.needs_processing


def test_a_settings_change_invalidates_the_frame(tmp_path, make_raw_tree, params):
    """The rule the old flow was missing entirely."""
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"
    process(processed, raw_path, params)

    changed = PipelineParams(camera=CameraSettings(saturation=11765))
    frame = worklist.decide(raw_path, processed, changed)

    assert frame.reason is Reason.PARAMS_CHANGED
    assert frame.needs_processing


def test_a_failed_frame_is_picked_up_again(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"
    process(processed, raw_path, params, status=ImageStatus.ERROR)

    frame = worklist.decide(raw_path, processed, params)

    assert frame.reason is Reason.PRIOR_ERROR
    assert frame.needs_processing


def test_a_frame_that_never_finished_is_picked_up_again(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"
    process(processed, raw_path, params, status=ImageStatus.RECEIVED)

    frame = worklist.decide(raw_path, processed, params)

    assert frame.reason is Reason.INCOMPLETE


def test_forcing_overrides_a_matching_fingerprint(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"
    process(processed, raw_path, params)

    frame = worklist.decide(raw_path, processed, params, force_new=True)

    assert frame.reason is Reason.FORCED
    assert frame.needs_processing


def test_an_unreadable_document_is_treated_as_missing(tmp_path, make_raw_tree, params):
    """One truncated file must not stop a walk over half a million frames."""
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"
    written = process(processed, raw_path, params)
    written["metadata"].write_text("{ this is not json")

    frame = worklist.decide(raw_path, processed, params)

    assert frame.reason is Reason.MISSING


def test_a_document_without_a_status_is_reprocessed(tmp_path, make_raw_tree, params):
    """Documents predating status recording must not be stranded."""
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"
    header = fits.getheader(raw_path)
    path_info = ImagePathInfo.from_fits_header(header)
    products.write_frame(processed, path_info, extract_metadata(header, path_info))

    frame = worklist.decide(raw_path, processed, params)

    assert frame.status is ImageStatus.UNKNOWN
    assert frame.needs_processing


@pytest.mark.parametrize(
    "content",
    [
        "{ this is not json",
        "[]",
        "42",
        '"a string"',
        "null",
        '{"image": null}',
        '{"image": []}',
        '{"image": {"status": []}}',
        '{"image": {"status": 7}}',
        '{"image": {"status": "NOT_A_STATUS"}}',
    ],
    ids=[
        "truncated",
        "array",
        "number",
        "string",
        "null",
        "null-image",
        "array-image",
        "unhashable-status",
        "numeric-status",
        "unknown-status",
    ],
)
def test_a_malformed_document_never_stops_the_walk(tmp_path, make_raw_tree, params, content):
    """Valid JSON is not a valid document; one bad file must not abort a walk."""
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"
    written = process(processed, raw_path, params)
    written["metadata"].write_text(content)

    frame = worklist.decide(raw_path, processed, params)

    assert frame.needs_processing
    assert len(worklist.build(raw_root, processed, params)) == 1


def test_a_document_that_is_not_utf8_is_treated_as_missing(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"
    written = process(processed, raw_path, params)
    written["metadata"].write_bytes(b"\xff\xfe not text at all")

    assert worklist.decide(raw_path, processed, params).reason is Reason.MISSING


# --- identity -------------------------------------------------------------


def test_identity_comes_from_the_path_without_reading_the_file(tmp_path):
    """The archive layout already says where a frame belongs."""
    path = tmp_path / "PAN001" / "abc123" / "20220115T082108" / "20220115T082209.fits"
    path.parent.mkdir(parents=True)

    path_info = worklist.identify(path)

    assert not path.exists()
    assert path_info.unit_id == "PAN001"
    assert path_info.camera_id == "abc123"
    assert path_info.image_id == "PAN001_abc123_20220115T082209"


def test_identity_falls_back_to_the_header_for_a_flat_layout(tmp_path, raw_frame):
    import shutil

    flat = tmp_path / "20160909T081314.fits"
    shutil.copy(raw_frame, flat)

    assert worklist.identify(flat).unit_id == "PAN001"


# --- the walk -------------------------------------------------------------


def test_the_walk_finds_every_frame(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=3)

    frames = worklist.build(raw_root, tmp_path / "processed", params)

    assert len(frames) == 3
    assert all(frame.reason is Reason.MISSING for frame in frames)


def test_the_walk_ignores_files_that_are_not_frames(tmp_path, make_raw_tree, params):
    """A stray `wcs.fits` or solver leftover is not a frame."""
    raw_root = make_raw_tree(count=2)
    (raw_root / "wcs.fits").write_bytes(b"")
    (raw_root / "20220115T082200-back.fits").write_bytes(b"")

    assert len(worklist.build(raw_root, tmp_path / "processed", params)) == 2


def test_the_walk_is_ordered_so_it_can_be_diffed(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=3)

    first = worklist.build(raw_root, tmp_path / "processed", params)
    second = worklist.build(raw_root, tmp_path / "processed", params)

    assert [f.raw_path for f in first] == [f.raw_path for f in second]
    assert [f.raw_path for f in first] == sorted(f.raw_path for f in first)


def test_a_frame_that_cannot_be_placed_is_skipped_not_fatal(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=2)
    broken = raw_root / "20220115T083000.fits"
    broken.write_bytes(b"not a fits file")

    frames = worklist.build(raw_root, tmp_path / "processed", params)

    assert len(frames) == 2
    assert broken not in [frame.raw_path for frame in frames]


def test_pending_is_only_the_work(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=3)
    processed = tmp_path / "processed"
    for raw_path in list(worklist.find_frames(raw_root))[:2]:
        process(processed, raw_path, params)

    frames = worklist.build(raw_root, processed, params)

    assert len(frames) == 3
    assert len(worklist.pending(frames)) == 1


def test_a_settings_change_invalidates_exactly_what_was_expected(tmp_path, make_raw_tree, params):
    """The diff this whole artifact exists to make possible."""
    raw_root = make_raw_tree(count=3)
    processed = tmp_path / "processed"
    for raw_path in worklist.find_frames(raw_root):
        process(processed, raw_path, params)

    before = worklist.build(raw_root, processed, params)
    after = worklist.build(
        raw_root, processed, PipelineParams(camera=CameraSettings(saturation=11765))
    )

    assert worklist.summarize(before) == {"up to date": 3}
    assert worklist.summarize(after) == {"params changed": 3}


# --- the artifact ---------------------------------------------------------


def test_the_table_has_a_row_per_frame_with_its_reason(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=3)

    table = worklist.as_table(worklist.build(raw_root, tmp_path / "processed", params))

    assert isinstance(table, pandas.DataFrame)
    assert len(table) == 3
    assert set(table.reason) == {"missing"}
    assert table.needs_processing.all()


def test_the_table_round_trips_through_csv(tmp_path, make_raw_tree, params):
    """It is only a diffable artifact if it survives being written out."""
    raw_root = make_raw_tree(count=2)
    table = worklist.as_table(worklist.build(raw_root, tmp_path / "processed", params))

    path = tmp_path / "worklist.csv"
    table.to_csv(path, index=False)

    assert pandas.read_csv(path).image_id.tolist() == table.image_id.tolist()


def test_an_empty_walk_still_produces_a_table(tmp_path, params):
    table = worklist.as_table(worklist.build(tmp_path / "nothing", tmp_path / "processed", params))

    assert table.empty
    assert "reason" in table.columns


def test_a_custom_metadata_filename_is_respected(tmp_path, make_raw_tree, params):
    raw_root = make_raw_tree(count=1)
    raw_path = next(worklist.find_frames(raw_root))
    processed = tmp_path / "processed"
    process(processed, raw_path, params)

    frame = worklist.decide(
        raw_path, processed, params, files=FileSettings(metadata_filename="elsewhere.json")
    )

    assert frame.reason is Reason.MISSING


# --- the statuses ---------------------------------------------------------


def test_error_sorts_below_processing():
    """So a failed frame compares as unprocessed and is picked up again."""
    assert ImageStatus.ERROR < ImageStatus.PROCESSING
    assert ImageStatus.UNKNOWN < ImageStatus.PROCESSING
    assert ImageStatus.MATCHED > ImageStatus.PROCESSING


def test_the_recorded_document_carries_the_settings_that_made_it(raw_header, raw_path_info, params):
    metadata = products.record_processing(
        extract_metadata(raw_header, raw_path_info), params, ImageStatus.MATCHED
    )

    assert metadata["image"]["params_fingerprint"] == params.fingerprint
    assert metadata["image"]["status"] == "MATCHED"
    assert metadata["image"]["params"]["camera"]["saturation"] == params.camera.saturation


def test_a_recorded_document_still_passes_the_store_check(raw_header, raw_path_info, params):
    products.check_document(
        products.record_processing(
            extract_metadata(raw_header, raw_path_info), params, ImageStatus.MATCHED
        )
    )
