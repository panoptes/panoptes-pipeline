"""Decide which frames need processing, and why, without opening any pixels.

Three walks exist over the archive and only one of them is expensive.
Distinguishing them matters because two are seconds and the third is hours:
the index walk reads documents that already exist (#185), the work-list walk
is this module, and processing is the per-frame work that consumes what this
module emits.

The work list is deliberately **an inspectable artifact rather than a loop
buried inside a batch command**. Over 563,566 archive frames the scope and the
reasons are worth seeing before committing to hours of work, and a diffable
work list is how a parameter change proves it invalidated what was expected
and nothing more. `as_table` is what makes that diff possible.

No pixel data is read. The decision is made from the frame's identity, whether
a document exists, and two fields inside it. Identity comes from the path where
the raw tree is laid out the way the archive is, and only falls back to reading
the FITS *header* when it is not -- so the walk is a directory scan in the
normal case and never more than a header read in the worst one. See data
contract 5.1.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas
from astropy.time import Time
from dateutil.parser import parse as parse_date
from loguru import logger
from panoptes.utils.images import fits as fits_utils
from panoptes.utils.images.fits import ImagePathInfo

from panoptes.pipeline import products
from panoptes.pipeline.settings import FileSettings, PipelineParams
from panoptes.pipeline.status import ImageStatus, image_status

#: Only match a properly named frame: `20220115T082209.fits`, optionally
#: compressed. Carried over unchanged from the deleted `observation.py`, where
#: it filtered a bucket listing rather than a directory walk. It is what keeps
#: a stray `wcs.fits` or a solver leftover out of the work list.
FITS_MATCHER = re.compile(r".*/\d{8}T\d{6}\.fits(\.fz)?$")


class Reason(StrEnum):
    """Why a frame is, or is not, in the work list.

    The four processing reasons are the ones data contract 5.2 names.
    `UP_TO_DATE` is the fifth because a work list that shows only the work
    cannot be diffed against one taken before a settings change -- the frames
    that *stopped* needing work are as informative as the ones that started.
    """

    #: No document at all. Never processed, or the output root is new.
    MISSING = "missing"
    #: A document exists but was produced by different settings.
    PARAMS_CHANGED = "params changed"
    #: The last attempt failed.
    PRIOR_ERROR = "prior error"
    #: A document exists but the frame never got past `PROCESSING`.
    INCOMPLETE = "incomplete"
    #: The caller asked for everything regardless.
    FORCED = "forced"
    #: Done, at these settings. Not work.
    UP_TO_DATE = "up to date"


@dataclass(frozen=True)
class Frame:
    """One frame's verdict, and enough context to act on it or read it."""

    raw_path: Path
    path_info: ImagePathInfo
    reason: Reason
    status: ImageStatus
    fingerprint: str | None = None

    @property
    def needs_processing(self) -> bool:
        return self.reason is not Reason.UP_TO_DATE


def identify(raw_path: Path) -> ImagePathInfo:
    """Work out where a raw frame belongs in the output tree.

    The path is tried first: the raw tree mirrors the bucket layout, so
    ``PAN001/abc123/20220115T082108/20220115T082209.fits`` already says
    everything needed and costs no I/O. Only a path that does not parse falls
    back to the header, which is what happens for a flat directory of frames.
    """
    try:
        return ImagePathInfo(path=str(raw_path))
    except ValueError:
        pass

    # `from_fits_header` reads `FILENAME` first and only catches `ValueError`,
    # so a frame without that keyword raises `KeyError` rather than falling
    # through to `SEQID`/`IMAGEID`. Most POCS frames carry it; a frame that does
    # not is still perfectly identifiable.
    header = fits_utils.getheader(str(raw_path))
    try:
        return ImagePathInfo.from_fits_header(header)
    except KeyError:
        unit_id, camera_id, sequence_time = header["SEQID"].split("_")
        _, _, image_time = header["IMAGEID"].split("_")
        return ImagePathInfo(
            unit_id=unit_id,
            camera_id=camera_id,
            sequence_time=Time(parse_date(sequence_time)),
            image_time=Time(parse_date(image_time)),
        )


def decide(
    raw_path: Path,
    processed_root: Path | str,
    params: PipelineParams,
    *,
    force_new: bool = False,
    files: FileSettings | None = None,
    path_info: ImagePathInfo | None = None,
) -> Frame:
    """Decide whether `raw_path` needs processing, and say why.

    The rules, in the order data contract 5.2 states them:

    1. No ``metadata.json`` -- process.
    2. Present, status at or past `PROCESSING`, fingerprint matches -- skip.
    3. Present, fingerprint differs, or status is an error -- reprocess.
    4. Forced -- always reprocess.

    Rule 3 is the one the old flow was missing. It stored `params` in the
    document and never compared them, so a settings change silently left stale
    products in place and nothing downstream could tell.
    """
    files = files or FileSettings()
    path_info = path_info or identify(raw_path)

    document_path = products.frame_directory(processed_root, path_info) / files.metadata_filename
    document = products.read_document(document_path)

    if document is None:
        # Forced still reports MISSING: there was nothing to force past.
        return Frame(raw_path, path_info, Reason.MISSING, ImageStatus.UNKNOWN)

    # The document may not have been written by this pipeline, so nothing about
    # its shape can be assumed. An `image` that is absent, null or not a map
    # leaves the frame looking unprocessed, which is the safe direction.
    image = document.get("image")
    image = image if isinstance(image, dict) else {}

    status = image_status(image.get("status"))
    fingerprint = image.get("params_fingerprint")

    if force_new:
        reason = Reason.FORCED
    elif status is ImageStatus.ERROR:
        reason = Reason.PRIOR_ERROR
    elif status < ImageStatus.PROCESSING:
        reason = Reason.INCOMPLETE
    elif fingerprint != params.fingerprint:
        reason = Reason.PARAMS_CHANGED
    else:
        reason = Reason.UP_TO_DATE

    return Frame(raw_path, path_info, reason, status, fingerprint)


def find_frames(raw_root: Path | str) -> Iterator[Path]:
    """Every properly named FITS frame under `raw_root`, in a stable order.

    Sorted because a work list that reorders between runs cannot be diffed,
    which is most of the point of having one.
    """
    raw_root = Path(raw_root)
    return iter(sorted(p for p in raw_root.rglob("*.fits*") if FITS_MATCHER.match(p.as_posix())))


def build(
    raw_root: Path | str,
    processed_root: Path | str,
    params: PipelineParams | None = None,
    *,
    force_new: bool = False,
    files: FileSettings | None = None,
) -> list[Frame]:
    """Classify every frame under `raw_root` against `processed_root`.

    Returns every frame, including the ones needing no work; filter with
    `pending` or read the whole thing with `as_table`. A frame whose path or
    header cannot be parsed into an `ImagePathInfo` is logged and left out
    rather than stopping the walk -- it is not a frame this pipeline can place
    in the output tree, and one bad file should not cost a walk over the
    archive.
    """
    params = params or PipelineParams()

    frames = []
    for raw_path in find_frames(raw_root):
        try:
            frames.append(
                decide(raw_path, processed_root, params, force_new=force_new, files=files)
            )
        except (ValueError, KeyError, OSError) as e:
            logger.warning(f"Skipping {raw_path}: cannot determine where it belongs ({e!r})")

    return frames


def pending(frames: Iterable[Frame]) -> list[Frame]:
    """Just the frames that need processing."""
    return [frame for frame in frames if frame.needs_processing]


def as_table(frames: Iterable[Frame]) -> pandas.DataFrame:
    """The work list as a table, for writing to CSV and diffing.

    One row per frame, sorted by image id, with the reason spelled out. This
    is the artifact: ``build(...)`` then ``as_table(...).to_csv(...)`` before
    and after a settings change shows exactly which frames the change
    invalidated.
    """
    return pandas.DataFrame(
        [
            dict(
                image_id=frame.path_info.image_id,
                unit_id=frame.path_info.unit_id,
                camera_id=frame.path_info.camera_id,
                sequence_id=frame.path_info.sequence_id,
                reason=str(frame.reason),
                status=frame.status.name,
                fingerprint=frame.fingerprint,
                needs_processing=frame.needs_processing,
                raw_path=str(frame.raw_path),
            )
            for frame in frames
        ],
        columns=[
            "image_id",
            "unit_id",
            "camera_id",
            "sequence_id",
            "reason",
            "status",
            "fingerprint",
            "needs_processing",
            "raw_path",
        ],
    )


def summarize(frames: Iterable[Frame]) -> dict[str, int]:
    """How many frames fell into each reason, for a one-line log."""
    counts: dict[str, int] = {}
    for frame in frames:
        counts[str(frame.reason)] = counts.get(str(frame.reason), 0) + 1
    return dict(sorted(counts.items()))
