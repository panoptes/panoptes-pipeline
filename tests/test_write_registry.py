"""`scripts/write_registry.py`: measurements in, registry profiles out (data contract 6.2)."""

import csv
import json
import sys
from pathlib import Path

import pytest

from panoptes.pipeline.registry import CameraProfile
from panoptes.pipeline.settings import CameraSettings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import write_registry  # noqa: E402

TABLE = Path(__file__).resolve().parents[1] / "plans" / "figures" / "gain_units.csv"


@pytest.fixture(scope="module")
def rows() -> dict[str, dict[str, str]]:
    with TABLE.open() as f:
        return {f"{r['unit']}/{r['camera']}": r for r in csv.DictReader(f)}


@pytest.fixture(scope="module")
def written(rows) -> dict[str, CameraProfile]:
    """Every row of the committed table, as the profile the script writes."""
    return {key: CameraProfile.model_validate(write_registry.profile(row)) for key, row in rows.items()}


def test_every_row_makes_a_valid_profile(written):
    assert len(written) == 34


def test_a_scaled_raw_gets_black_and_a_warning_only(written, rows):
    """A 2048-black body clipping at 16383 is PAN019's scaled raw, whatever the unit."""
    scaled = {
        key
        for key, row in rows.items()
        if write_registry.is_scaled(int(row["black_assumed"]), int(row["clip"]))
    }
    assert scaled == {k for k in rows if k.startswith("PAN019/")}

    for key in scaled:
        profile = written[key]
        assert profile.warning
        assert profile.black_level.value == 2048
        assert (profile.clip, profile.gain, profile.read_noise) == (None, None, None)


def test_a_512_black_body_at_16383_is_not_scaled(written):
    """PAN007's newer bodies clip at the full 14 bits legitimately."""
    profile = written["PAN007/d37295"]

    assert profile.warning is None
    assert (profile.black_level.value, profile.clip.value) == (512, 16383)


def test_no_profile_carries_a_fleet_wide_default(written):
    """``params_camera_*`` is never a profile source, so its values must not appear."""
    defaults = CameraSettings()
    for key, profile in written.items():
        if profile.gain is not None:
            assert profile.gain.value != defaults.effective_gain, key
        if profile.clip is not None:
            assert profile.clip.value != defaults.saturation, key


def test_a_rounded_clip_is_written_as_the_clip(rows, written):
    assert rows["PAN018/54d3e9"]["clip"] == "13584"

    assert written["PAN018/54d3e9"].clip.value == 13583


def test_gain_is_the_family_value_with_the_cameras_own_fit_in_the_note(written):
    profile = written["PAN025/df3495"]  # its own cloudy fit reads 1.40-1.74

    assert (profile.gain.value, profile.gain.uncertainty, profile.gain.method) == (2.1, 0.2, "family")
    assert "own shared fit 1.40-1.74" in profile.gain.note


def test_a_sensor_with_no_family_value_gets_no_gain(rows):
    row = dict(rows["PAN012/358d0f"], sensor="9999x9999")

    assert "gain" not in write_registry.profile(row)


def test_a_non_positive_read_noise_is_absent_and_explained(rows, written):
    assert float(rows["PAN008/62b062"]["read_noise_e"]) == 0.0
    missing = sorted(k for k, p in written.items() if p.read_noise is None and not p.warning)

    assert missing == ["PAN008/62b062"]
    assert "No read noise" in written["PAN008/62b062"].note


def test_a_single_sequence_is_flagged_as_weak(written):
    assert "weak" in written["PAN018/798cf1"].gain.note
    assert "weak" not in written["PAN010/45fdcf"].gain.note  # two sequences


def test_an_existing_profile_that_differs_is_left_alone(tmp_path, rows):
    """The tree is state people edit by hand; a regenerated note must not clobber theirs."""
    document = write_registry.profile(rows["PAN012/358d0f"])
    assert write_registry.write(document, tmp_path).startswith("wrote")
    path = tmp_path / "units" / "PAN012" / "cameras" / "358d0f.json"
    edited = dict(json.loads(path.read_text()), note="measured on flats 2027-01-01")
    path.write_text(json.dumps(edited))

    assert write_registry.write(document, tmp_path).startswith("DIFFERS")
    assert json.loads(path.read_text())["note"] == "measured on flats 2027-01-01"
    assert write_registry.write(document, tmp_path, force=True).startswith("wrote")
    assert "note" not in json.loads(path.read_text())


def test_an_identical_profile_is_reported_unchanged(tmp_path, rows):
    document = write_registry.profile(rows["PAN012/358d0f"])
    write_registry.write(document, tmp_path)

    assert write_registry.write(document, tmp_path).startswith("unchanged")
