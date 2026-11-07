"""The camera registry: per camera uid, what no header carries (data contract 6.2)."""

import json

import numpy as np
import pytest
from astropy.io import fits
from panoptes.utils.images.fits import ImagePathInfo

from panoptes.pipeline import processing, products, provenance, registry, worklist
from panoptes.pipeline.processing import calibrate
from panoptes.pipeline.provenance import Provenance
from panoptes.pipeline.settings import CameraSettings, PipelineParams
from panoptes.pipeline.status import ImageStatus
from panoptes.pipeline.utils.images import extract_metadata
from panoptes.pipeline.worklist import Reason

UNIT, CAMERA = "PAN001", "abc123"


def write_profile(root, unit_id=UNIT, camera_uid=CAMERA, **values):
    """A profile with values chosen to match no header and no default."""
    document = dict(
        unit_id=unit_id,
        camera_uid=camera_uid,
        iso=100,
        black_level=dict(value=2048, method="body"),
        clip=dict(value=13583, method="histogram"),
        gain=dict(value=2.3, uncertainty=0.2, method="family"),
        read_noise=dict(value=17.6, method="sky-ptc"),
    )
    document.update(values)
    path = registry.profile_path(unit_id, camera_uid, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2))
    return path


@pytest.fixture
def registry_root(tmp_path):
    root = tmp_path / "registry"
    write_profile(root)
    return root


@pytest.fixture
def header(raw_header):
    """The tiny POCS frame, relabeled as the camera the test registry knows."""
    header = raw_header.copy()
    header["SEQID"] = f"{UNIT}_{CAMERA}_20220115T082108"
    header["IMAGEID"] = f"{UNIT}_{CAMERA}_20220115T082200"
    return header


# --- loading --------------------------------------------------------------


def test_an_unknown_camera_has_no_profile(registry_root):
    assert registry.load_profile(UNIT, "ffffff", registry_root) is None
    assert registry.fingerprint(UNIT, "ffffff", registry_root) is None


def test_a_malformed_profile_raises_rather_than_falling_through(registry_root):
    path = write_profile(registry_root, gain=dict(value=2.3, method="guessed"))
    assert path.is_file()

    with pytest.raises(registry.RegistryError, match="is not a valid camera profile"):
        registry.load_profile(UNIT, CAMERA, registry_root)


def test_a_profile_filed_under_the_wrong_camera_raises(registry_root):
    path = write_profile(registry_root, camera_uid="def456")
    path.rename(registry.profile_path(UNIT, "fff000", registry_root))

    with pytest.raises(registry.RegistryError, match="describes PAN001/def456"):
        registry.load_profile(UNIT, "fff000", registry_root)


def test_an_edit_is_seen_at_once_despite_the_cache(registry_root):
    before = registry.load_profile(UNIT, CAMERA, registry_root)
    write_profile(registry_root, gain=dict(value=2.45, method="flat-ptc"))

    after = registry.load_profile(UNIT, CAMERA, registry_root)

    assert before.gain.value == 2.3
    assert after.gain.value == 2.45


def test_the_fingerprint_ignores_formatting_and_follows_content(registry_root):
    path = registry.profile_path(UNIT, CAMERA, registry_root)
    before = registry.fingerprint(UNIT, CAMERA, registry_root)

    path.write_text(json.dumps(json.loads(path.read_text())))  # no indent
    reformatted = registry.fingerprint(UNIT, CAMERA, registry_root)
    write_profile(registry_root, clip=dict(value=13584, method="histogram"))
    edited = registry.fingerprint(UNIT, CAMERA, registry_root)

    assert reformatted == before
    assert edited != before


def test_the_header_identity_is_the_shared_one(header, registry_root):
    """`worklist.identify_header`, i.e. `panoptes-utils`' `from_fits_header`, not a local parse."""
    settings = CameraSettings(registry_dir=registry_root)
    assert "FILENAME" in header  # the shared parser tries it first and falls back to SEQID
    assert worklist.identify_header(header).camera_id == CAMERA

    assert provenance.camera_profile(header, settings) == registry.load_profile(UNIT, CAMERA, registry_root)


def test_a_placeholder_uid_has_no_profile_and_builds_no_path(raw_header, registry_root, monkeypatch):
    """`tiny.fits` is a real POCS frame whose uid is ``XXXXXX``, outside the shared grammar."""
    assert worklist.identify_header(raw_header).camera_id == "XXXXXX"
    assert registry.identifier_problem("PAN001", "XXXXXX")
    monkeypatch.setattr(registry, "profile_path", lambda *a, **k: pytest.fail("built a path"))

    profile = provenance.camera_profile(raw_header, CameraSettings(registry_dir=registry_root))

    assert profile is None


def test_a_header_that_identifies_nothing_has_no_profile(header, registry_root):
    del header["FILENAME"], header["SEQID"]

    assert provenance.camera_profile(header, CameraSettings(registry_dir=registry_root)) is None


def test_with_no_registry_configured_a_helper_resolves_no_profile(header):
    assert CameraSettings().registry_dir is None

    resolved = provenance.resolve_camera(header, CameraSettings())

    assert resolved["zero_bias"].provenance is Provenance.DEFAULT


def test_loading_from_no_registry_raises():
    with pytest.raises(registry.RegistryError, match="no camera registry configured"):
        registry.load_profile(UNIT, CAMERA, None)


@pytest.mark.parametrize("entry", ["frame", "observation"])
def test_processing_refuses_to_start_without_a_registry(tmp_path, make_raw_tree, monkeypatch, entry):
    """Without one, every frame would be calibrated with fleet-wide defaults."""
    monkeypatch.setattr(processing, "require_solver", lambda: None)
    monkeypatch.setattr(processing.fits_utils, "getdata", lambda *a, **k: pytest.fail("read pixels"))
    raw_root = make_raw_tree(count=1)
    params = PipelineParams()
    assert params.camera.registry_dir is None

    with pytest.raises(registry.RegistryError, match="no camera registry configured"):
        if entry == "frame":
            processing.process_frame(next(worklist.find_frames(raw_root)), tmp_path / "out", params)
        else:
            processing.process_observation(raw_root, tmp_path / "out", params)


def test_processing_refuses_a_registry_that_is_not_a_directory(tmp_path, make_raw_tree, monkeypatch):
    monkeypatch.setattr(processing, "require_solver", lambda: None)
    params = PipelineParams(camera=CameraSettings(registry_dir=tmp_path / "nowhere"))

    with pytest.raises(registry.RegistryError, match="is not a directory"):
        processing.process_frame(next(worklist.find_frames(make_raw_tree(count=1))), tmp_path / "out", params)


# --- resolution -----------------------------------------------------------


def test_the_registry_resolves_what_no_header_carries(header, registry_root):
    settings = CameraSettings(registry_dir=registry_root)
    assert header["WHTLVLN"] == 11765  # the header tier is there to be outranked
    assert "EGAIN" not in header

    resolved = provenance.resolve_camera(header, settings)

    source = f"units/{UNIT}/cameras/{CAMERA}"
    assert resolved["zero_bias"].value == 2048
    assert resolved["zero_bias"].source == f"{source}.black_level"
    assert resolved["saturation"].value == 13583
    assert resolved["saturation"].source == f"{source}.clip"
    assert resolved["effective_gain"].value == 2.3
    assert resolved["effective_gain"].uncertainty == 0.2
    assert resolved["read_noise"].value == 17.6
    for name in ("zero_bias", "saturation", "effective_gain", "read_noise"):
        assert resolved[name].provenance is Provenance.REGISTRY, name


def test_the_registry_gain_outranks_the_pocs_class_constant(header, registry_root):
    """``EGAIN`` is POCS's hardcoded 1.5 (data contract 6.3), not a measurement."""
    header["EGAIN"] = 1.5

    resolved = provenance.resolve_camera(header, CameraSettings(registry_dir=registry_root))

    assert resolved["effective_gain"].value == 2.3


def test_without_a_profile_the_old_tiers_still_apply(header, tmp_path):
    resolved = provenance.resolve_camera(header, CameraSettings(registry_dir=tmp_path / "empty"))

    assert resolved["saturation"].source == "WHTLVLN"
    assert resolved["zero_bias"].provenance is Provenance.DEFAULT
    assert resolved["read_noise"].provenance is Provenance.DEFAULT
    assert resolved["read_noise"].value is None


def test_an_iso_mismatch_keeps_only_the_black_level(header, registry_root):
    header["ISO"] = 400

    resolved = provenance.resolve_camera(header, CameraSettings(registry_dir=registry_root))

    assert resolved["zero_bias"].provenance is Provenance.REGISTRY
    assert resolved["saturation"].source == "WHTLVLN"
    assert resolved["effective_gain"].provenance is Provenance.DEFAULT
    assert resolved["read_noise"].provenance is Provenance.DEFAULT


def test_an_explicit_camera_outranks_the_header(header, registry_root):
    write_profile(registry_root, camera_uid="def456", black_level=dict(value=512, method="body"))
    settings = CameraSettings(registry_dir=registry_root)
    assert provenance.resolve_camera(header, settings)["zero_bias"].value == 2048

    resolved = provenance.resolve_camera(header, settings, camera=(UNIT, "def456"))

    assert resolved["zero_bias"].value == 512


def test_calibrate_subtracts_the_registry_black_and_masks_at_the_clip(header, registry_root):
    params = PipelineParams(camera=CameraSettings(registry_dir=registry_root))
    raw = np.full((40, 40), 2148, dtype="uint16")
    raw[0, 0] = 13583  # the clip
    raw[0, 2] = 12000  # above WHTLVLN, below the clip: real signal

    calibrated = calibrate(raw, header, params)

    assert calibrated.calibration["zero_bias"].value == 2048
    assert calibrated.mask[0, 0]
    assert not calibrated.mask[0, 2]


def test_a_real_frame_without_filename_still_finds_its_profile(widefield, widefield_registry):
    """`widefield.fits.fz` has no ``FILENAME``, which `from_fits_header` alone cannot handle."""
    header = fits.getheader(widefield, ext=1)
    assert "FILENAME" not in header
    assert worklist.identify_header(header).camera_id == "df3495"

    resolved = provenance.resolve_camera(header, CameraSettings(registry_dir=widefield_registry))

    assert resolved["zero_bias"].value == 2048
    assert resolved["zero_bias"].provenance is Provenance.REGISTRY
    assert resolved["saturation"].value == 13583


# --- the fingerprint in the document and the work list --------------------


def test_the_registry_location_is_not_fingerprinted(tmp_path):
    moved = PipelineParams(camera=CameraSettings(registry_dir=tmp_path))

    assert moved.fingerprint == PipelineParams().fingerprint


def process(root, raw_path, params):
    header = fits.getheader(raw_path)
    path_info = ImagePathInfo.from_fits_header(header)
    metadata = extract_metadata(header, path_info, params.camera)
    metadata = products.record_processing(metadata, params, ImageStatus.MATCHED)
    return products.write_frame(root, path_info, metadata), metadata


def test_the_document_records_the_profile_it_was_resolved_against(tmp_path, make_raw_tree, registry_root):
    params = PipelineParams(camera=CameraSettings(registry_dir=registry_root))
    raw_path = next(worklist.find_frames(make_raw_tree(count=1)))

    _, metadata = process(tmp_path / "processed", raw_path, params)

    assert metadata["image"]["registry_fingerprint"] == registry.fingerprint(UNIT, CAMERA, registry_root)
    assert metadata["image"]["calibration"]["zero_bias"]["provenance"] == "registry"


def test_a_registry_edit_invalidates_that_camera_and_no_other(tmp_path, make_raw_tree, registry_root):
    params = PipelineParams(camera=CameraSettings(registry_dir=registry_root))
    write_profile(registry_root, camera_uid="def456")
    make_raw_tree(count=1, camera_id=CAMERA)
    raw_root = make_raw_tree(count=1, camera_id="def456")
    processed = tmp_path / "processed"
    for raw_path in worklist.find_frames(raw_root):
        process(processed, raw_path, params)
    assert {f.reason for f in worklist.build(raw_root, processed, params)} == {Reason.UP_TO_DATE}

    write_profile(registry_root, gain=dict(value=2.45, method="flat-ptc"))
    after = {f.path_info.camera_id: f.reason for f in worklist.build(raw_root, processed, params)}

    assert after == {CAMERA: Reason.REGISTRY_CHANGED, "def456": Reason.UP_TO_DATE}


def test_a_new_profile_invalidates_frames_resolved_without_one(tmp_path, make_raw_tree):
    root = tmp_path / "registry"
    params = PipelineParams(camera=CameraSettings(registry_dir=root))
    raw_root = make_raw_tree(count=1)
    processed = tmp_path / "processed"
    _, metadata = process(processed, next(worklist.find_frames(raw_root)), params)
    assert metadata["image"]["registry_fingerprint"] is None

    write_profile(root)
    (frame,) = worklist.build(raw_root, processed, params)

    assert frame.reason is Reason.REGISTRY_CHANGED


# --- review fixes ---------------------------------------------------------


@pytest.mark.parametrize(
    "unit_id, camera_uid",
    [
        ("../../outside", CAMERA),
        (UNIT, "../../x"),
        ("/abs", CAMERA),
        (UNIT, "abc12"),
        ("", CAMERA),
        ("PAN01", CAMERA),
        ("x/PAN001", CAMERA),
        (UNIT, "XXXXXX"),
    ],
)
def test_an_unsafe_identifier_never_reaches_the_filesystem(tmp_path, unit_id, camera_uid):
    """A crafted path or header component must not read a file outside the root."""
    root = tmp_path / "registry"
    root.mkdir()
    outside = tmp_path / "outside" / "cameras" / f"{CAMERA}.json"
    outside.parent.mkdir(parents=True)
    outside.write_text(json.dumps(dict(unit_id="../../outside", camera_uid=CAMERA)))

    with pytest.raises(registry.RegistryError):
        registry.load_profile(unit_id, camera_uid, root)


def test_a_malformed_profile_stops_the_walk_instead_of_dropping_the_camera(
    tmp_path, make_raw_tree, registry_root
):
    """`build` skips frames it cannot place, on `ValueError`; a bad profile is not that."""
    params = PipelineParams(camera=CameraSettings(registry_dir=registry_root))
    raw_root = make_raw_tree(count=2)
    processed = tmp_path / "processed"
    for raw_path in worklist.find_frames(raw_root):
        process(processed, raw_path, params)
    write_profile(registry_root, gain=dict(value=2.3, method="guessed"))

    with pytest.raises(registry.RegistryError, match="units/PAN001/cameras/abc123.json"):
        worklist.build(raw_root, processed, params)


@pytest.mark.parametrize(
    "field, value",
    [("gain", 0), ("gain", -2.3), ("clip", -1), ("black_level", -2048), ("read_noise", -5)],
)
def test_an_unphysical_value_is_refused(registry_root, field, value):
    write_profile(registry_root, **{field: dict(value=value, method="body")})

    with pytest.raises(registry.RegistryError):
        registry.load_profile(UNIT, CAMERA, registry_root)


def test_a_negative_uncertainty_is_refused(registry_root):
    write_profile(registry_root, gain=dict(value=2.3, uncertainty=-0.2, method="family"))

    with pytest.raises(registry.RegistryError):
        registry.load_profile(UNIT, CAMERA, registry_root)


def test_the_work_list_reads_each_profile_once_per_walk(tmp_path, make_raw_tree, registry_root, monkeypatch):
    params = PipelineParams(camera=CameraSettings(registry_dir=registry_root))
    raw_root = make_raw_tree(count=5)
    processed = tmp_path / "processed"
    for raw_path in worklist.find_frames(raw_root):
        process(processed, raw_path, params)

    calls = []
    real = registry.load_profile
    monkeypatch.setattr(registry, "load_profile", lambda *a, **k: calls.append(a) or real(*a, **k))
    frames = worklist.build(raw_root, processed, params)

    assert len(frames) == 5
    assert {f.reason for f in frames} == {Reason.UP_TO_DATE}
    assert len(calls) == 1


def test_the_document_records_the_profile_it_was_given(header, raw_path_info, registry_root):
    """One profile per frame: the one the pixels were calibrated with is the one recorded.

    Loading it twice -- once to calibrate, once to write the document -- lets an
    edit in between record values that were never applied.
    """
    settings = CameraSettings(registry_dir=registry_root)
    applied = registry.load_profile(UNIT, CAMERA, registry_root)
    write_profile(registry_root, black_level=dict(value=512, method="body"))  # edited after calibrating
    path_info = ImagePathInfo(path=f"{UNIT}/{CAMERA}/20220115T082108/20220115T082200.fits")

    metadata = extract_metadata(header, path_info, settings, profile=applied)

    assert metadata["image"]["calibration"]["zero_bias"]["value"] == 2048
    assert metadata["image"]["registry_fingerprint"] == applied.fingerprint
    assert applied.fingerprint != registry.fingerprint(UNIT, CAMERA, registry_root)


def test_calibrate_reports_the_profile_it_used(header, registry_root):
    params = PipelineParams(camera=CameraSettings(registry_dir=registry_root))

    calibrated = calibrate(np.full((40, 40), 2148, dtype="uint16"), header, params)

    assert calibrated.profile == registry.load_profile(UNIT, CAMERA, registry_root)
