"""The document must survive a round trip to a document store untransformed.

That is the property that makes re-attaching one later a walk-and-upload
rather than a migration, so it is checked before the write rather than at
upload time. See data contract 3.2 and 3.3.
"""

import json
from datetime import UTC, datetime

import numpy as np
import pandas
import pytest
from astropy.io import fits
from astropy.time import Time

from panoptes.pipeline import processing, products
from panoptes.pipeline.products import DocumentError
from panoptes.pipeline.utils.images import extract_metadata


def test_the_frame_directory_mirrors_the_bucket_layout(tmp_path, raw_path_info):
    directory = products.frame_directory(tmp_path, raw_path_info)

    assert directory.parent.parent.parent.name == raw_path_info.unit_id
    assert directory.parent.parent.name == raw_path_info.camera_id
    assert directory.relative_to(tmp_path).parts == (
        raw_path_info.unit_id,
        raw_path_info.camera_id,
        directory.parent.name,
        directory.name,
    )


def test_writing_a_frame_produces_the_named_artifacts(tmp_path, raw_path_info, raw_header):
    metadata = extract_metadata(raw_header, raw_path_info)
    sources = pandas.DataFrame({"picid": [1, 2], "x": [3.0, 4.0]})

    written = products.write_frame(
        tmp_path,
        raw_path_info,
        metadata,
        reduced=np.zeros((4, 4), dtype=np.float32),
        background=np.ones((2, 2), dtype=np.float32),
        rms=np.ones((2, 2), dtype=np.float32),
        mask=np.zeros((4, 4), dtype=bool),
        sources=sources,
        header=raw_header,
    )

    assert set(written) == {"metadata", "reduced", "sources"}
    for path in written.values():
        assert path.exists()
    assert {p.name for p in products.frame_directory(tmp_path, raw_path_info).iterdir()} == {
        "metadata.json",
        "image.fits",
        "sources.parquet",
    }
    assert pandas.read_parquet(written["sources"]).equals(sources)


def test_the_pixels_are_one_file_with_named_extensions(tmp_path):
    """One file, because reduced and background always change together."""
    path = products.write_image(
        tmp_path / "image.fits",
        np.zeros((8, 8), dtype=np.float32),
        background=np.ones((2, 2), dtype=np.float32),
        rms=np.full((2, 2), 3.0, dtype=np.float32),
        mask=np.zeros((8, 8), dtype=bool),
    )

    planes = processing.read_image(path)

    assert set(planes) == {"reduced", "background", "rms", "mask"}
    assert planes["reduced"].shape == (8, 8)
    assert planes["mask"].shape == (8, 8)


def test_the_background_is_stored_as_a_mesh_not_a_full_frame(tmp_path):
    """The model is four orders of magnitude smaller than its interpolation."""
    reduced = np.zeros((400, 400), dtype=np.float32)
    mesh = np.ones((3, 5, 5), dtype=np.float32)

    path = products.write_image(tmp_path / "image.fits", reduced, background=mesh, rms=mesh)
    planes = processing.read_image(path)

    assert planes["background"].shape == mesh.shape
    assert planes["background"].shape != reduced.shape

    with fits.open(path) as hdul:
        assert hdul["BACKGROUND"].header["MESH"] is True


def test_raw_pixels_are_not_written(tmp_path):
    """The raw frame already exists upstream; copying it buys nothing."""
    path = products.write_image(tmp_path / "image.fits", np.zeros((8, 8), dtype=np.float32))

    assert "raw" not in processing.read_image(path)


def test_a_masked_array_keeps_its_mask_as_nan(tmp_path):
    reduced = np.ma.array(np.ones((4, 4), dtype=np.float32), mask=np.eye(4, dtype=bool))

    planes = processing.read_image(
        products.write_image(tmp_path / "image.fits", reduced, mask=np.eye(4, dtype=bool))
    )

    assert np.isnan(planes["reduced"]).sum() == 4
    assert planes["mask"].sum() == 4


def test_an_existing_image_is_kept_unless_forced(tmp_path):
    products.write_image(tmp_path / "image.fits", np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(FileExistsError):
        products.write_image(tmp_path / "image.fits", np.zeros((4, 4), dtype=np.float32), force_new=False)


def test_only_the_document_is_required(tmp_path, raw_path_info, raw_header):
    """A caller that has not solved yet writes what it has, not nulls."""
    written = products.write_frame(tmp_path, raw_path_info, extract_metadata(raw_header, raw_path_info))

    assert set(written) == {"metadata"}


def test_the_document_is_valid_json_with_iso_times(tmp_path, raw_path_info, raw_header):
    metadata = extract_metadata(raw_header, raw_path_info)
    path = products.write_frame(tmp_path, raw_path_info, metadata)["metadata"]

    document = json.loads(path.read_text())

    # Parsing it back as a date is the check: a `datetime` repr would not.
    datetime.fromisoformat(document["image"]["image_time"])
    datetime.fromisoformat(document["sequence"]["sequence_time"])


def test_the_camera_identifiers_are_in_the_image_document(raw_header, raw_path_info):
    """They used to sit only in the sequence, which made a per-frame join awkward."""
    metadata = extract_metadata(raw_header, raw_path_info)

    assert metadata["image"]["camera"]["camera_id"] == raw_path_info.camera_id
    assert metadata["image"]["camera"]["serial_number"] == str(raw_header["CAMSN"])
    assert metadata["image"]["camera"]["serial_number"] == (metadata["sequence"]["camera"]["serial_number"])


def test_an_absent_serial_is_null_not_the_string_none(raw_header, raw_path_info):
    del raw_header["CAMSN"]

    metadata = extract_metadata(raw_header, raw_path_info)

    assert metadata["image"]["camera"]["serial_number"] is None
    assert metadata["sequence"]["camera"]["serial_number"] != "None"


def test_the_document_records_calibration_provenance(raw_header, raw_path_info):
    calibration = extract_metadata(raw_header, raw_path_info)["image"]["calibration"]

    assert calibration["saturation"]["provenance"] == "header"
    assert calibration["saturation"]["source"] == "WHTLVLN"
    assert calibration["zero_bias"]["provenance"] == "default"


def test_dimensions_are_not_silently_zero_on_a_raw_frame(raw_header, raw_path_info):
    metadata = extract_metadata(raw_header, raw_path_info)

    assert metadata["sequence"]["imagew"] == raw_header["NAXIS1"]
    assert metadata["sequence"]["imagew"] != 0


def test_a_dotted_field_name_is_refused():
    with pytest.raises(DocumentError, match="cannot contain"):
        products.check_document({"camera.serial_number": "012070048413"})


def test_a_nested_dotted_field_name_is_refused():
    with pytest.raises(DocumentError, match="cannot contain"):
        products.check_document({"image": {"camera": {"white.level": 11765}}})


def test_arrays_of_arrays_are_refused():
    with pytest.raises(DocumentError, match="arrays of arrays"):
        products.check_document({"stamp": [[1, 2], [3, 4]]})


def test_a_bulk_table_in_the_document_is_refused():
    with pytest.raises(DocumentError, match="parquet"):
        products.check_document({"sources": pandas.DataFrame({"picid": [1]})})


def test_a_pixel_array_in_the_document_is_refused():
    with pytest.raises(DocumentError, match="beside the document"):
        products.check_document({"background": np.zeros((4, 4))})


def test_a_real_document_passes_the_check(raw_header, raw_path_info):
    products.check_document(extract_metadata(raw_header, raw_path_info))


def test_times_encode_as_iso_8601():
    moment = datetime(2016, 9, 9, 8, 12, 26, tzinfo=UTC)

    assert products.encode_value(moment).startswith("2016-09-09T08:12:26")
    assert products.encode_value(Time(moment)).startswith("2016-09-09T08:12:26")
    assert products.encode_value(np.float32(1.5)) == 1.5


def test_an_unserializable_value_raises_rather_than_being_dropped():
    with pytest.raises(TypeError, match="Cannot serialize"):
        products.encode_value(object())


def test_a_failed_check_writes_nothing(tmp_path, raw_path_info):
    with pytest.raises(DocumentError):
        products.write_frame(tmp_path, raw_path_info, {"bad.key": 1})

    assert not products.frame_directory(tmp_path, raw_path_info).joinpath("metadata.json").exists()


def test_an_existing_document_is_kept_unless_forced(tmp_path, raw_path_info, raw_header):
    metadata = extract_metadata(raw_header, raw_path_info)
    products.write_frame(tmp_path, raw_path_info, metadata)

    with pytest.raises(FileExistsError):
        products.write_frame(tmp_path, raw_path_info, metadata, force_new=False)

    products.write_frame(tmp_path, raw_path_info, metadata, force_new=True)


def test_no_temporary_file_is_left_behind(tmp_path, raw_path_info, raw_header):
    products.write_frame(tmp_path, raw_path_info, extract_metadata(raw_header, raw_path_info))

    directory = products.frame_directory(tmp_path, raw_path_info)
    assert not list(directory.glob("*.tmp*"))


def test_the_sequence_document_is_not_written_here(tmp_path, raw_path_info, raw_header):
    """It is the one file two frame workers would contend on; it belongs to #184."""
    products.write_frame(tmp_path, raw_path_info, extract_metadata(raw_header, raw_path_info))

    assert not list(tmp_path.rglob("observation.json"))
