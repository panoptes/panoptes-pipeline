"""A resolved value must say where it came from.

The defect these guard against is not a crash. It is that "saturation was
11435, read from ``WHTLVLN``" and "saturation was 15872, because nobody knew"
used to serialize identically, so no reader could tell a measurement from a
fleet-wide constant. See data contract 3.4.
"""

import pytest

from panoptes.pipeline import provenance
from panoptes.pipeline.provenance import Provenance
from panoptes.pipeline.settings import CameraSettings


def test_saturation_comes_from_the_header_not_the_default(raw_header):
    resolved = provenance.resolve_camera(raw_header, CameraSettings())
    saturation = resolved["saturation"]

    assert saturation.provenance is Provenance.HEADER
    assert saturation.source == "WHTLVLN"
    assert saturation.value == raw_header["WHTLVLN"]

    # The measured white level is well below the fleet-wide constant, which is
    # the whole point: at 15872 the pipeline fails to mask saturated pixels.
    assert saturation.value < CameraSettings().saturation


def test_a_default_is_labeled_as_one(bare_header):
    resolved = provenance.resolve_camera(bare_header, CameraSettings())

    assert resolved["saturation"].provenance is Provenance.DEFAULT
    assert resolved["saturation"].source == "camera.saturation"
    assert resolved["saturation"].value == CameraSettings().saturation
    assert resolved["saturation"].is_default


def test_zero_bias_is_always_a_default_today(raw_header):
    """No frame in the archive carries a black-level keyword."""
    resolved = provenance.resolve_camera(raw_header, CameraSettings())
    assert resolved["zero_bias"].provenance is Provenance.DEFAULT


def test_dimensions_come_from_naxis_not_imagew(raw_header):
    """`IMAGEW` is written by the plate solver, so a raw frame has none.

    Reading it gave a silent zero, which is why 16% of the observation index
    has null dimensions.
    """
    assert "IMAGEW" not in raw_header

    resolved = provenance.resolve_camera(raw_header, CameraSettings())

    assert resolved["image_width"].provenance is Provenance.HEADER
    assert resolved["image_width"].source == "NAXIS1"
    assert resolved["image_width"].value == raw_header["NAXIS1"]
    assert resolved["image_height"].value == raw_header["NAXIS2"]
    assert resolved["image_width"].value != 0


def test_dimensions_agree_between_a_raw_and_a_solved_frame(solved_header):
    """Where both keywords exist they must not disagree."""
    resolved = provenance.resolve_camera(solved_header, CameraSettings())

    assert resolved["image_width"].value == solved_header["IMAGEW"]
    assert resolved["image_height"].value == solved_header["IMAGEH"]


def test_an_empty_keyword_counts_as_absent():
    """POCS writes `SEQID = ''` on some frames; empty is missing, not a value."""
    from astropy.io import fits

    header = fits.Header({"FIELD": "   "})
    assert provenance.from_header(header, "FIELD", str) is None


def test_an_unparseable_keyword_falls_through_rather_than_raising():
    from astropy.io import fits

    header = fits.Header({"WHTLVLN": "not a number"})
    resolved = provenance.resolve_camera(header, CameraSettings())

    assert resolved["saturation"].provenance is Provenance.DEFAULT


def test_defaulted_names_what_nobody_knew(bare_header, raw_header):
    assert "saturation" in provenance.defaulted(provenance.resolve_camera(bare_header, CameraSettings()))
    assert "saturation" not in provenance.defaulted(provenance.resolve_camera(raw_header, CameraSettings()))


def test_the_document_form_is_plain_json_types(raw_header):
    document = provenance.as_document(provenance.resolve_camera(raw_header, CameraSettings()))

    assert document["saturation"]["provenance"] == "header"
    assert isinstance(document["saturation"]["provenance"], str)
    assert set(document["saturation"]) == {"value", "provenance", "source"}


@pytest.mark.parametrize("tier", list(Provenance))
def test_every_tier_serializes_as_a_bare_string(tier):
    assert isinstance(tier.value, str)
    assert f"{tier}" == tier.value
