"""Where a calibration value came from, recorded beside the value itself.

The old flow serialized a dump of global settings into every document, so
"saturation was 11435, read from ``WHTLVLN``" and "saturation was 15872,
because nobody knew" came out identical. That is how
``params_camera_saturation=15872`` came to sit in archive records against 37
different cameras: the pipeline faithfully recorded a fleet-wide default and
nothing downstream could tell it apart from a measurement.

A `Resolved` carries the value *and* the tier it came from, so the two cases
are distinguishable by anything reading the archive. See data contract 3.4.

The tiers are ordered best to worst. Nothing here refuses to produce a value:
declaring which header keywords are required, and failing when one is missing,
is a separate change against the header contract. What this module guarantees
is that a fallback is never silent -- it is labeled `Provenance.DEFAULT` and
carries the name of the setting it fell back to.

One rule follows for anything reading the archive, and it is worth stating
where the tiers are defined: **``params_camera_*`` in existing records is
never a source for a camera profile.** Those fields are the fleet-wide
defaults, so reading them back would close a loop in which the pipeline
rediscovers its own wrong constants.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from astropy.io.fits import Header
from loguru import logger
from pydantic import BaseModel

from panoptes.pipeline import registry
from panoptes.pipeline.registry import CameraProfile, Measurement
from panoptes.pipeline.settings import CameraSettings


class Provenance(StrEnum):
    """Where a resolved value came from, best tier first.

    `StrEnum` so the value serializes as a plain string with no encoder hook,
    which is what keeps the document loadable into a document store.
    """

    #: Read from a FITS header keyword written by POCS. Per frame, and free.
    HEADER = "header"
    #: Measured from this observation's own pixels.
    MEASURED = "measured"
    #: Looked up in the camera registry, keyed on the camera uid.
    REGISTRY = "registry"
    #: A fleet-wide default. Nobody knew; this is the tier that means "unknown".
    DEFAULT = "default"


class Resolved(BaseModel):
    """A calibration value together with the tier it was resolved from.

    ``source`` names the specific origin within the tier -- a header keyword
    (``"WHTLVLN"``), or the settings attribute a default came from
    (``"camera.saturation"``) -- so a record says not just how well a value is
    known but exactly where to look to check it.
    """

    value: float | int | str | None
    provenance: Provenance
    source: str | None = None
    #: One-sigma, where the source states one. Only the registry does today.
    uncertainty: float | None = None

    @property
    def is_default(self) -> bool:
        """True when nobody knew and a fleet-wide constant was used."""
        return self.provenance is Provenance.DEFAULT


def from_header(header: Header, keyword: str, cast: type = float) -> Resolved | None:
    """Resolve `keyword` from `header`, or return None if it is absent.

    Returns None rather than a `Resolved` with a null value, so a caller can
    write ``from_header(...) or fallback(...)`` and have the tiers fall through
    in order. An empty string counts as absent: POCS writes ``SEQID = ''`` on
    some frames, and an empty keyword is missing data rather than a value.
    """
    raw = header.get(keyword)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None

    try:
        value = cast(raw)
    except (TypeError, ValueError):
        return None

    return Resolved(value=value, provenance=Provenance.HEADER, source=keyword)


def from_default(value: float | int | str | None, source: str) -> Resolved:
    """Label `value` as a fleet-wide default that nobody could do better than."""
    return Resolved(value=value, provenance=Provenance.DEFAULT, source=source)


def from_registry(profile: CameraProfile | None, name: str) -> Resolved | None:
    """Resolve the registry value `name` from `profile`, or None if it has none.

    ``source`` is the profile's document path and field,
    ``units/PAN012/cameras/358d0f.gain``; the method and its caveats are in the
    profile itself, whose fingerprint the frame document records.
    """
    if profile is None:
        return None
    measurement: Measurement | None = getattr(profile, name)
    if measurement is None:
        return None
    return Resolved(
        value=measurement.value,
        provenance=Provenance.REGISTRY,
        source=f"units/{profile.unit_id}/cameras/{profile.camera_uid}.{name}",
        uncertainty=measurement.uncertainty,
    )


def camera_profile(
    header: Header,
    settings: CameraSettings,
    camera: tuple[str, str] | None = None,
) -> CameraProfile | None:
    """The registry profile for the frame's camera, or None if there is none.

    `camera` is ``(unit_id, camera_uid)``; without it the identity is read from
    ``SEQID``/``IMAGEID``. Pass it where the path is authoritative, as it is in
    `processing.process_frame`.
    """
    camera = camera or registry.camera_from_header(header)
    if camera is None:
        return None
    return registry.load_profile(*camera, root=settings.registry_dir)


def resolve_camera(
    header: Header,
    settings: CameraSettings,
    camera: tuple[str, str] | None = None,
) -> dict[str, Resolved]:
    """Resolve the per-frame camera calibration, each value with its tier.

    `camera` is ``(unit_id, camera_uid)``, the registry key; see
    `camera_profile`. The values are known to very different degrees:

    ``saturation``
        The registry's measured **clip** first: the raw value the histogram
        piles up at, 13583 on a 2048-black Canon body at ISO 100. Then
        ``WHTLVLN``, which is on every frame back to 2016 but is Canon's
        rendering white point, not the clip -- 11765 against 13583 on the
        paper's camera -- so it masks 10--13% too low. Too low is the safe
        direction, which is why it stays as a fallback; the fleet-wide default
        of 16384 masks nothing at all. The header tier is ranked below the
        registry here because the keyword does not measure what is being
        resolved (data contract 6.1).

    ``image_width`` / ``image_height``
        ``NAXIS1``/``NAXIS2``, which are in every FITS file by definition.
        The old code read ``IMAGEW``/``IMAGEH``, which POCS never writes --
        astrometry.net adds them during plate solving, so they are present on
        solved frames and absent on raw ones. ``int(header.get("IMAGEW", 0))``
        then yields a silent zero, which is why 16% of the observation index
        has null dimensions.

    ``effective_gain``
        The registry, then ``EGAIN``, then a default. The registry outranks the
        header because ``EGAIN``, where POCS writes it at all, is the Canon
        class constant 1.5 (data contract 6.3), which no camera in the archive
        measures. Once panoptes/POCS#1481 writes a per-body value, a camera
        with no profile picks it up here unchanged.

    ``zero_bias``
        The registry's black level. No header carries it, so without a
        profile this is a default -- and the default of 512 is wrong by 1536
        ADU for every 2048-black body in the archive.

    ``read_noise``
        The registry, or unknown. There is no default to fall back to, and
        none is invented.

    Clip, gain and read noise depend on ISO, so they are taken from the
    registry only when the frame's ``ISO`` matches the profile's. Black does
    not.
    """
    profile = camera_profile(header, settings, camera)
    iso_profile = profile
    if profile is not None:
        if profile.warning:
            logger.warning(f"{profile.unit_id}/{profile.camera_uid}: {profile.warning}")
        iso = header.get("ISO")
        if iso is not None and int(iso) != profile.iso:
            logger.warning(
                f"{profile.unit_id}/{profile.camera_uid}: frame is ISO {iso}, registry profile is "
                f"ISO {profile.iso}; not using its clip, gain or read noise"
            )
            iso_profile = None

    saturation = (
        from_registry(iso_profile, "clip")
        or from_header(header, "WHTLVLN", int)
        or from_default(settings.saturation, "camera.saturation")
    )

    # NAXIS1/NAXIS2 are mandatory in a FITS file, so the default is unreachable
    # on a real frame. It exists so a truncated header degrades loudly-in-the-
    # record rather than to a zero that looks like a measurement.
    width = from_header(header, "NAXIS1", int) or from_default(settings.image_width, "camera.image_width")
    height = from_header(header, "NAXIS2", int) or from_default(settings.image_height, "camera.image_height")

    gain = (
        from_registry(iso_profile, "gain")
        or from_header(header, "EGAIN", float)
        or from_default(settings.effective_gain, "camera.effective_gain")
    )

    zero_bias = from_registry(profile, "black_level") or from_default(settings.zero_bias, "camera.zero_bias")

    read_noise = from_registry(iso_profile, "read_noise") or from_default(None, "unknown")

    return dict(
        saturation=saturation,
        image_width=width,
        image_height=height,
        effective_gain=gain,
        zero_bias=zero_bias,
        read_noise=read_noise,
    )


def as_document(resolved: dict[str, Resolved]) -> dict[str, Any]:
    """Render resolved values as plain nested maps for the metadata document."""
    return {name: value.model_dump(mode="json") for name, value in resolved.items()}


def defaulted(resolved: dict[str, Resolved]) -> list[str]:
    """Names of the values that fell back to a fleet-wide default.

    What a caller warns about, and what a survey of the archive counts.
    """
    return sorted(name for name, value in resolved.items() if value.is_default)
