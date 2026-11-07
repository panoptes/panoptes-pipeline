"""The camera registry: what no header carries, per camera uid.

Data contract 6.2. One document per camera, at
``units/{unit_id}/cameras/{camera_uid}.json`` under a registry root -- the path
a document store would use, readable offline, no new mechanism. Each holds
four values, the ones no raw frame carries and every frame needs:

``black_level``
    Raw ADU. A property of the body: 2048 on the EOS 100D and 24 MP classes,
    512 on PAN007's newer bodies. No legacy header records it and the archive
    holds no bias frames, so it is established camera by camera.
``clip``
    Raw ADU where the detector saturates, read off the raw histogram. *Not*
    ``WHTLVLN``, which is Canon's rendering white point and sits 10--13% below
    it on a 100D-class body.
``gain``
    e-/ADU at the profile's ISO.
``read_noise``
    e-, at the profile's ISO.

The key is the **camera uid** -- the six characters in ``INSTRUME`` and in
every ``sequence_id`` -- never ``CAMSN``, which current POCS no longer writes,
and never the ``Cam00`` slot name (data contract 2.3).

**``params_camera_*`` in existing archive records is never a profile source.**
Those fields are this pipeline's own fleet-wide defaults, recorded as
provenance; reading them back would close a loop in which the pipeline
rediscovers its own wrong constants. Every value here names the method that
produced it.

The registry shipped with the package lives in ``data/registry`` beside this
module, so a release says exactly which calibration it carries.
``CameraSettings.registry_dir`` points somewhere else. A profile that exists but
does not validate raises `RegistryError`: a typo in a calibration file is not
the same thing as an unknown camera, and must not fall through to a default.
`RegistryError` is deliberately not a `ValueError`, which is what the work-list
walk catches to skip a frame it cannot place; a bad profile has to stop the
walk, not quietly drop every frame of that camera from it.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum, StrEnum
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from astropy.io.fits import Header
from panoptes.utils.error import PanError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

#: The registry that ships with the package.
PACKAGED_ROOT = Path(str(files("panoptes.pipeline") / "data" / "registry"))

#: Characters of a profile's digest kept as its fingerprint, as for params.
FINGERPRINT_LENGTH = 12

#: What a unit id or camera uid may contain. Both become path components, and
#: both reach here from headers and paths this pipeline did not write, so
#: anything that could leave the registry root -- ``..``, a separator, an empty
#: string -- is refused before a path is built.
IDENTIFIER = re.compile(r"[A-Za-z0-9]+")

#: The uid is the first six characters of the gphoto serial (data contract 2.3).
CAMERA_UID_LENGTH = 6


class RegistryError(PanError):
    """A registry profile, or a request for one, that cannot be trusted.

    A `PanError` and not a `ValueError` on purpose; see the module docstring.
    """


class Lookup(Enum):
    """Sentinel: find the profile, as opposed to "this camera has none" (None)."""

    LOOKUP = "lookup"


#: Pass as ``profile=`` to look the profile up rather than supply it.
LOOKUP = Lookup.LOOKUP


class Method(StrEnum):
    """How a registry value was established. Recorded so a reader can weigh it."""

    #: A known property of the body model, checked against the camera's own data
    #: (for black, the free intercept of its photon transfer curve).
    BODY = "body"
    #: Read off the raw pixel histogram: the value the counts pile up at.
    HISTOGRAM = "histogram"
    #: This camera's own photon transfer on science frames. A lower bound on the
    #: gain: cloud, twilight and drift over unresolved stars only add variance.
    SKY_PTC = "sky-ptc"
    #: The value for the camera's sensor family, from sky photon transfer across
    #: every camera in it. Used where the per-camera spread is the method's own
    #: ~10% floor rather than a difference between bodies.
    FAMILY = "family"
    #: Photon transfer on flat fields: no stars, no drift. What settles the gain.
    FLAT_PTC = "flat-ptc"


class Measurement(BaseModel):
    """One registry value, with how it was established and how well."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    uncertainty: float | None = Field(default=None, ge=0)
    method: Method
    note: str | None = None


class CameraProfile(BaseModel):
    """Everything the registry knows about one camera.

    A value that is not known is absent, never a placeholder: an absent value
    falls through to the next tier with its provenance saying so.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str
    camera_uid: str
    #: ``CAMSN``, the number on the body, as a string: it carries leading zeros.
    #: A cross-reference for people, never a key.
    serial_number: str | None = None
    #: ``NAXIS1 x NAXIS2``, which identifies the sensor family.
    sensor: str | None = None
    #: The ISO the gain, read noise and clip were measured at. Black does not
    #: depend on it.
    iso: int = 100
    black_level: Measurement | None = None
    clip: Measurement | None = None
    gain: Measurement | None = None
    read_noise: Measurement | None = None
    #: Logged whenever the profile is used. For a camera whose frames need care
    #: before anything is trusted.
    warning: str | None = None
    #: Anything a reader of the profile should know that belongs to no one value,
    #: such as why a value is absent. Not logged.
    note: str | None = None

    @model_validator(mode="after")
    def _identity(self) -> CameraProfile:
        if problem := identifier_problem(self.unit_id, self.camera_uid):
            raise ValueError(problem)
        return self

    @model_validator(mode="after")
    def _physical(self) -> CameraProfile:
        """Refuse values no detector can have, rather than calibrating with them."""
        for name, positive in (("gain", True), ("clip", True), ("black_level", False), ("read_noise", False)):
            measurement = getattr(self, name)
            if measurement is None:
                continue
            if measurement.value < 0 or (positive and measurement.value == 0):
                bound = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be {bound}, not {measurement.value}")
        return self

    @property
    def fingerprint(self) -> str:
        """A digest of the profile's content, independent of file formatting.

        Recorded in every frame document resolved against this profile, so the
        work-list walk can tell when a registry edit has invalidated a frame --
        and invalidate only that camera's frames.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:FINGERPRINT_LENGTH]


def identifier_problem(unit_id: str, camera_uid: str) -> str | None:
    """What is wrong with this pair as registry identifiers, or None if nothing."""
    for name, value in (("unit_id", unit_id), ("camera_uid", camera_uid)):
        if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
            return f"{name} {value!r} is not a registry identifier (letters and digits only)"
    if len(camera_uid) != CAMERA_UID_LENGTH:
        return f"camera_uid {camera_uid!r} is not {CAMERA_UID_LENGTH} characters (data contract 2.3)"
    return None


def check_identifiers(unit_id: str, camera_uid: str) -> None:
    """Raise `RegistryError` unless both are safe, single path components."""
    if problem := identifier_problem(unit_id, camera_uid):
        raise RegistryError(problem)


def profile_path(unit_id: str, camera_uid: str, root: Path | str | None = None) -> Path:
    """Where the profile for `camera_uid` on `unit_id` lives under `root`.

    Raises `RegistryError` for an identifier that could leave `root`.
    """
    check_identifiers(unit_id, camera_uid)
    root = Path(root) if root is not None else PACKAGED_ROOT
    return root / "units" / unit_id / "cameras" / f"{camera_uid}.json"


def load_profile(unit_id: str, camera_uid: str, root: Path | str | None = None) -> CameraProfile | None:
    """The registry's profile for this camera, or None if it has none.

    None only when there is no file. A file that does not parse or validate, or
    that names a different camera from the path it sits at, raises
    `RegistryError` naming the file.
    """
    path = profile_path(unit_id, camera_uid, root)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None

    try:
        profile = _parse(text)
    except ValidationError as error:
        raise RegistryError(f"{path} is not a valid camera profile: {error}") from error
    if (profile.unit_id, profile.camera_uid) != (unit_id, camera_uid):
        raise RegistryError(
            f"{path} describes {profile.unit_id}/{profile.camera_uid}, not {unit_id}/{camera_uid}"
        )
    return profile


@lru_cache(maxsize=256)
def _parse(text: str) -> CameraProfile:
    """Validate a profile once per distinct content.

    The work-list walk asks for a camera's profile once per frame, over half a
    million frames. Keyed on the text rather than the modification time, which
    has a coarse enough granularity that a same-size edit can look unchanged.
    """
    return CameraProfile.model_validate_json(text)


def fingerprint(unit_id: str, camera_uid: str, root: Path | str | None = None) -> str | None:
    """The fingerprint of this camera's profile, or None if it has none."""
    profile = load_profile(unit_id, camera_uid, root)
    return profile.fingerprint if profile is not None else None


def profiles(root: Path | str | None = None) -> list[CameraProfile]:
    """Every profile under `root`, sorted by unit and camera."""
    root = Path(root) if root is not None else PACKAGED_ROOT
    return [
        load_profile(path.parent.parent.name, path.stem, root)
        for path in sorted(root.glob("units/*/cameras/*.json"))
    ]


def camera_from_header(header: Header) -> tuple[str, str] | None:
    """``(unit_id, camera_uid)`` from ``SEQID``, then ``IMAGEID``, or None.

    Both are ``<unit>_<uid>_<time>``. An empty keyword counts as absent: POCS
    writes ``SEQID = ''`` on some frames.
    """
    for keyword in ("SEQID", "IMAGEID"):
        value = header.get(keyword)
        if isinstance(value, str) and value.count("_") >= 2:
            unit_id, camera_uid, *_ = value.split("_")
            if unit_id and camera_uid:
                return unit_id, camera_uid
    return None
