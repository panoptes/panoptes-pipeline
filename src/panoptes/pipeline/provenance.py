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
from pydantic import BaseModel

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
    #: Looked up in the camera registry, keyed on the body serial.
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


def resolve_camera(header: Header, settings: CameraSettings) -> dict[str, Resolved]:
    """Resolve the per-frame camera calibration, each value with its tier.

    Four values, and they are known to very different degrees:

    ``saturation``
        ``WHTLVLN`` gives the white level per frame, offline, on every frame
        in the archive back to 2016. Measured values run 11435--12277 against
        a fleet-wide default of 15872, so both cameras checked in data
        contract 2.1 sit roughly 25% below the hardcoded threshold and the
        default fails to mask saturated pixels on either. Free, and worth
        taking.

    ``image_width`` / ``image_height``
        ``NAXIS1``/``NAXIS2``, which are in every FITS file by definition.
        The old code read ``IMAGEW``/``IMAGEH``, which POCS never writes --
        astrometry.net adds them during plate solving, so they are present on
        solved frames and absent on raw ones. ``int(header.get("IMAGEW", 0))``
        then yields a silent zero, which is why 16% of the observation index
        has null dimensions.

    ``effective_gain``
        ``EGAIN`` where POCS wrote it, which is rare. Otherwise a default,
        until the camera registry can answer it by body serial.

    ``zero_bias``
        Not in the header at all, on any frame, so this is always a default
        today. It is load-bearing content for the registry.
    """
    saturation = from_header(header, "WHTLVLN", int) or from_default(
        settings.saturation, "camera.saturation"
    )

    # NAXIS1/NAXIS2 are mandatory in a FITS file, so the default is unreachable
    # on a real frame. It exists so a truncated header degrades loudly-in-the-
    # record rather than to a zero that looks like a measurement.
    width = from_header(header, "NAXIS1", int) or from_default(
        settings.image_width, "camera.image_width"
    )
    height = from_header(header, "NAXIS2", int) or from_default(
        settings.image_height, "camera.image_height"
    )

    gain = from_header(header, "EGAIN", float) or from_default(
        settings.effective_gain, "camera.effective_gain"
    )

    zero_bias = from_default(settings.zero_bias, "camera.zero_bias")

    return dict(
        saturation=saturation,
        image_width=width,
        image_height=height,
        effective_gain=gain,
        zero_bias=zero_bias,
    )


def as_document(resolved: dict[str, Resolved]) -> dict[str, Any]:
    """Render resolved values as plain nested maps for the metadata document."""
    return {name: value.model_dump(mode="json") for name, value in resolved.items()}


def defaulted(resolved: dict[str, Resolved]) -> list[str]:
    """Names of the values that fell back to a fleet-wide default.

    What a caller warns about, and what a survey of the archive counts.
    """
    return sorted(name for name, value in resolved.items() if value.is_default)
