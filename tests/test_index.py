"""The index must be deletable, rebuildable, and give the same answers.

That rule is what keeps a derived index from quietly becoming a second source
of truth, which is exactly how `observations.csv` ended up the only holder of
numbers nothing else records. Everything else here is in service of it.

These build their own processed tree from the committed POCS frames. The index
walk reads documents, and a document can be written without solving anything,
so none of this needs `solve-field`, a catalog, or the network.
"""

import pandas
import pytest
from astropy.io import fits
from panoptes.utils.images.fits import ImagePathInfo

from panoptes.pipeline import index, products
from panoptes.pipeline.settings import FileSettings, PipelineParams
from panoptes.pipeline.status import ImageStatus
from panoptes.pipeline.utils.images import extract_metadata


@pytest.fixture
def params() -> PipelineParams:
    return PipelineParams()


@pytest.fixture
def processed_tree(tmp_path, make_raw_tree, params):
    """A processed tree: real documents, written the way the pipeline writes them."""

    def build(count=3, status=ImageStatus.MATCHED, root_name="processed", **kwargs):
        raw_root = make_raw_tree(count=count, **kwargs)
        processed = tmp_path / root_name

        for raw_path in sorted(raw_root.rglob("*.fits")):
            header = fits.getheader(raw_path)
            path_info = ImagePathInfo.from_fits_header(header)
            metadata = products.record_processing(
                extract_metadata(header, path_info), params, status
            )
            metadata["image"]["sources"] = dict(num_detected=1000, photutils_fwhm_median=2.4)
            products.write_frame(processed, path_info, metadata)

        return processed

    return build


# --- the rule ------------------------------------------------------------


def test_the_index_can_be_deleted_and_rebuilt_identically(processed_tree, tmp_path):
    """The whole point. If this fails, something lives only in the index."""
    processed = processed_tree(count=3)
    written = index.build(processed, tmp_path / "index")
    before_frames, before_observations = index.read(tmp_path / "index")

    for path in written.values():
        path.unlink()
    assert not any(p.exists() for p in written.values())

    index.build(processed, tmp_path / "index")
    after_frames, after_observations = index.read(tmp_path / "index")

    pandas.testing.assert_frame_equal(before_frames, after_frames)
    pandas.testing.assert_frame_equal(before_observations, after_observations)


def test_the_observation_index_is_derived_from_the_frame_index(processed_tree, tmp_path):
    """Not built from the tree, so the two cannot disagree."""
    processed = processed_tree(count=3)
    frames = index.build_frames(processed)

    from_frames = index.build_observations(frames)
    from_rebuilt = index.build_observations(index.build_frames(processed))

    pandas.testing.assert_frame_equal(from_frames, from_rebuilt)
    assert from_frames.num_frames.iloc[0] == len(frames)


# --- the frame index -----------------------------------------------------


def test_one_row_per_frame(processed_tree):
    frames = index.build_frames(processed_tree(count=4))

    assert len(frames) == 4
    assert frames.image_uid.nunique() == 4


def test_nested_maps_become_flat_columns(processed_tree):
    frames = index.build_frames(processed_tree(count=1))

    assert "image_camera_white_lvln" in frames.columns
    assert "image_calibration_saturation_value" in frames.columns
    assert "image_calibration_saturation_provenance" in frames.columns


def test_no_column_name_contains_a_dot(processed_tree):
    """A dotted name is a view over a nested map, never storage."""
    frames = index.build_frames(processed_tree(count=1))

    assert not [c for c in frames.columns if "." in c]


def test_the_params_dump_is_dropped_but_its_fingerprint_is_kept(processed_tree, params):
    """`params_fingerprint` is a sibling of `params`, not inside it."""
    frames = index.build_frames(processed_tree(count=1))

    assert "image_params_camera_saturation" not in frames.columns
    assert "image_params_background_box_size" not in frames.columns
    assert "image_params_catalog_vmag_limits" not in frames.columns
    assert frames.image_params_fingerprint.iloc[0] == params.fingerprint


def test_a_document_that_cannot_be_read_is_skipped_not_fatal(processed_tree, tmp_path):
    processed = processed_tree(count=3)
    next(index.find_documents(processed)).write_text("{ not json")

    frames = index.build_frames(processed)

    assert len(frames) == 2


def test_a_missing_field_becomes_a_null_column(processed_tree):
    """Documents written by an older pipeline still have to index."""
    processed = processed_tree(count=2)
    paths = list(index.find_documents(processed))
    document = products.read_document(paths[0])
    del document["image"]["sources"]
    products.write_document(paths[0], document)

    frames = index.build_frames(processed)

    assert len(frames) == 2
    assert frames.image_sources_num_detected.isna().sum() == 1


def test_an_empty_tree_produces_an_empty_index(tmp_path):
    frames = index.build_frames(tmp_path / "nothing")
    observations = index.build_observations(frames)

    assert frames.empty
    assert observations.empty


def test_a_custom_metadata_filename_is_respected(processed_tree, tmp_path):
    processed = processed_tree(count=2)

    frames = index.build_frames(processed, files=FileSettings(metadata_filename="other.json"))

    assert frames.empty


def test_an_index_with_no_status_at_all_still_builds(processed_tree, tmp_path):
    """A tree written before statuses existed must index, not raise.

    Removing the field from one document leaves the column in place because
    another row supplies it, so the all-missing case needs its own test --
    that is how it was missed.
    """
    processed = processed_tree(count=2)
    for path in index.find_documents(processed):
        document = products.read_document(path)
        del document["image"]["status"]
        products.write_document(path, document)

    written = index.build(processed, tmp_path / "index")
    frames, observations = index.read(tmp_path / "index")

    assert written["frames"].exists()
    assert len(frames) == 2
    assert observations.num_usable.iloc[0] == 0


@pytest.mark.parametrize("field", ["status", "image_time", "uid", ["camera", "exptime"]])
def test_a_field_missing_from_every_document_does_not_raise(processed_tree, tmp_path, field):
    """Exercised through `build`, not just `build_frames` -- the aggregation is
    where a wholly absent column actually bites."""
    processed = processed_tree(count=2)
    for path in index.find_documents(processed):
        document = products.read_document(path)
        block = document["image"]
        keys = field if isinstance(field, list) else [field]
        for key in keys[:-1]:
            block = block[key]
        block.pop(keys[-1], None)
        products.write_document(path, document)

    index.build(processed, tmp_path / "index")


def test_an_empty_archive_has_the_same_schema_as_a_full_one(processed_tree, tmp_path):
    """A consumer needs one query schema that does not depend on the archive."""
    index.build(tmp_path / "nothing", tmp_path / "empty")
    index.build(processed_tree(count=2), tmp_path / "full")

    empty_frames, empty_obs = index.read(tmp_path / "empty")
    full_frames, full_obs = index.read(tmp_path / "full")

    assert list(empty_obs.columns) == list(full_obs.columns)
    assert set(index.REQUIRED_FRAME_COLUMNS) <= set(empty_frames.columns)
    assert set(index.REQUIRED_FRAME_COLUMNS) <= set(full_frames.columns)


# --- the schema manifest --------------------------------------------------


def test_the_schema_manifest_records_the_column_contract(processed_tree, tmp_path):
    """#207 requires the mapping be readable outside this package."""
    import json

    written = index.build(processed_tree(count=2), tmp_path / "index")
    manifest = json.loads(written["schema"].read_text())

    assert manifest["separator"] == "_"
    assert manifest["dropped"] == [["image", "params"]]
    assert manifest["files"]["observations.parquet"]["derived_from"] == "frames.parquet"
    assert manifest["files"]["frames.parquet"]["row"] == "frame"
    assert "image_calibration_saturation_value" in manifest["files"]["frames.parquet"]["columns"]


def test_the_schema_manifest_is_stable_across_rebuilds(processed_tree, tmp_path):
    processed = processed_tree(count=2)
    first = index.build(processed, tmp_path / "index")["schema"].read_text()
    second = index.build(processed, tmp_path / "index")["schema"].read_text()

    assert first == second


# --- the observation index -----------------------------------------------


def test_one_row_per_sequence(processed_tree):
    frames = index.build_frames(processed_tree(count=3))

    observations = index.build_observations(frames)

    assert len(observations) == 1
    assert observations.num_frames.iloc[0] == 3


def test_usable_frames_are_counted_separately_from_frames(processed_tree):
    """`ImageStatus` is per frame now, so "usable" is finally answerable."""
    processed = processed_tree(count=3)
    failed = list(index.find_documents(processed))[0]
    document = products.read_document(failed)
    document["image"]["status"] = ImageStatus.ERROR.name
    products.write_document(failed, document)

    observations = index.build_observations(index.build_frames(processed))

    assert observations.num_frames.iloc[0] == 3
    assert observations.num_usable.iloc[0] == 2


def test_total_exptime_is_a_sum_over_frames(processed_tree):
    """Not a number the index alone holds -- that is how it went null."""
    frames = index.build_frames(processed_tree(count=3))

    observations = index.build_observations(frames)

    assert observations.total_exptime.iloc[0] == pytest.approx(frames.image_camera_exptime.sum())


def test_duration_comes_from_the_frames(processed_tree):
    frames = index.build_frames(processed_tree(count=3))

    observations = index.build_observations(frames)

    assert observations.duration_minutes.iloc[0] > 0
    assert observations.start_time.iloc[0] < observations.end_time.iloc[0]


def test_a_consistent_camera_reports_one_serial(processed_tree):
    observations = index.build_observations(index.build_frames(processed_tree(count=3)))

    assert observations.num_serials.iloc[0] == 1


def test_an_inconsistent_serial_within_one_sequence_is_flagged(processed_tree):
    processed = processed_tree(count=3)
    paths = list(index.find_documents(processed))
    document = products.read_document(paths[0])
    document["sequence"]["camera"]["serial_number"] = "999999999999"
    products.write_document(paths[0], document)

    observations = index.build_observations(index.build_frames(processed))

    assert observations.num_serials.iloc[0] == 2


def test_a_serial_that_changes_between_sequences_is_flagged(processed_tree):
    """The defect data contract 2.3 actually describes.

    `14d3bd` carries 2,332 sequences on one serial and 4 on another. It is a
    property of the camera across sequences, so a per-sequence count cannot see
    it -- within any one sequence the serial is constant, and `num_serials` is 1
    for both.
    """
    processed = processed_tree(count=2, sequence_time="20220115T082108")
    processed_tree(count=2, sequence_time="20220115T090000", root_name="processed")

    for path in index.find_documents(processed):
        document = products.read_document(path)
        if "20220115T0900" in document["sequence"]["sequence_id"]:
            document["sequence"]["camera"]["serial_number"] = "999999999999"
            products.write_document(path, document)

    observations = index.build_observations(index.build_frames(processed))

    assert len(observations) == 2
    assert set(observations.num_serials) == {1}
    assert set(observations.camera_num_serials) == {2}


def test_a_sequence_with_no_exposure_recorded_stays_null(processed_tree):
    """Missing must not look like a measured zero."""
    processed = processed_tree(count=2)
    for path in index.find_documents(processed):
        document = products.read_document(path)
        del document["image"]["camera"]["exptime"]
        products.write_document(path, document)

    observations = index.build_observations(index.build_frames(processed))

    assert pandas.isna(observations.total_exptime.iloc[0])


# --- writing it out ------------------------------------------------------


def test_build_writes_both_files_and_the_schema(processed_tree, tmp_path):
    written = index.build(processed_tree(count=2), tmp_path / "index")

    assert set(written) == {"frames", "observations", "schema"}
    assert all(path.exists() for path in written.values())
    assert written["frames"].name == "frames.parquet"


def test_the_index_defaults_to_sitting_beside_what_it_describes(processed_tree):
    processed = processed_tree(count=2)

    written = index.build(processed)

    assert written["frames"].parent == processed


def test_the_written_index_round_trips(processed_tree, tmp_path):
    processed = processed_tree(count=3)
    index.build(processed, tmp_path / "index")

    frames, observations = index.read(tmp_path / "index")

    assert len(frames) == 3
    assert len(observations) == 1
