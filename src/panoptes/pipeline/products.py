"""Write the per-frame metadata document and products to a local tree.

The cloud removal deleted the Firestore writer, and `extract_metadata` was
left with nothing calling it -- so the pipeline produces no metadata at all
while everything downstream still reads products that are no longer being
made. This module closes that loop without Google Cloud.

Nothing here is new machinery. `extract_metadata` already returns
``dict(unit=..., sequence=..., image=...)``, and the deleted orchestrator read
exactly that from ``assets/metadata.json`` before writing each key to its own
Firestore document. **JSON was always the wire format and Firestore was a
sink.** What this module does is keep the file and drop the forward.

Layout
------
The directory tree mirrors the document path, which already mirrors the raw
bucket layout `ImagePathInfo` parses::

    <root>/PAN007/f6eb3d/20251127T044424/20251127T044520/
        metadata.json       # {unit, sequence, image}
        image.fits          # reduced
        extras.fits         # named ImageHDUs
        sources.parquet

So re-attaching a document store later is a walk-and-upload rather than a
migration, and `rsync` to the processing server needs no special handling.

The observation-level ``observation.json`` is deliberately not written here.
It is the one file two frame workers would contend on, and it belongs with the
processing entry points that know when a sequence is complete.

See data contract 3.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas
from astropy.io import fits
from astropy.time import Time
from loguru import logger
from panoptes.utils.images.fits import ImagePathInfo

from panoptes.pipeline.settings import FileSettings, PipelineParams
from panoptes.pipeline.status import ImageStatus

#: Field-name characters a document store will not accept. Firestore reads a
#: `.` as a path separator, so a key containing one is unaddressable rather
#: than merely ugly. The dotted `camera.serial_number` form in
#: `observations.csv` is a *view* over the nested map, never storage.
FORBIDDEN_IN_FIELD_NAMES = (".",)


class DocumentError(ValueError):
    """The document would not survive a round trip through a document store."""


def frame_directory(root: Path | str, path_info: ImagePathInfo) -> Path:
    """The directory holding one frame's products.

    ``<root>/{unit_id}/{camera_id}/{sequence_time}/{image_time}/``.

    `ImagePathInfo.as_path` produces the same hierarchy but with the image time
    as a *filename* stem rather than a directory, because it was built to name
    a single blob. A frame owns four files here, so it gets a directory.
    """
    return Path(root) / path_info.as_path()


def encode_value(value: Any) -> Any:
    """Convert one value into something `json` can write and a store can read.

    Times become ISO 8601 strings, which is the wire format the contract
    specifies and what a document store parses on ingest. Numpy scalars become
    Python scalars -- they arrive from every measured quantity in the pipeline
    and `json` cannot write them.
    """
    if isinstance(value, Time):
        return value.to_datetime().isoformat()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} into a metadata document")


def check_document(document: Mapping[str, Any], _path: str = "") -> None:
    """Raise `DocumentError` unless `document` is document-store compatible.

    Three rules, from data contract 3.2, and each has already been broken once:

    - Field names carry no ``.``.
    - No arrays of arrays. A document store has no nested-array type, so a list
      of lists silently loses its shape.
    - Bulk tables stay out of the document. A DataFrame or a large ndarray
      belongs beside it as parquet, which is what keeps the document far below
      the 1 MB Firestore limit.

    This runs before the write rather than at upload time, because the point of
    the local tree is that attaching a store later is an upload and not a
    migration. A document that fails this check would make it a migration.
    """
    for key, value in document.items():
        where = f"{_path}{key}"

        if not isinstance(key, str):
            raise DocumentError(f"{where!r}: field names must be strings")
        for char in FORBIDDEN_IN_FIELD_NAMES:
            if char in key:
                raise DocumentError(
                    f"{where!r}: field names cannot contain {char!r}; "
                    "nest the map instead of dotting the key"
                )

        if isinstance(value, pandas.DataFrame | pandas.Series):
            raise DocumentError(
                f"{where!r}: bulk tables belong beside the document as parquet, not in it"
            )

        if isinstance(value, Mapping):
            check_document(value, _path=f"{where}.")
        elif isinstance(value, np.ndarray):
            raise DocumentError(
                f"{where!r}: arrays belong beside the document, not in it (shape {value.shape})"
            )
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for index, item in enumerate(value):
                if isinstance(item, Sequence | np.ndarray) and not isinstance(item, str | bytes):
                    raise DocumentError(
                        f"{where}[{index}]: arrays of arrays have no document-store "
                        "equivalent; flatten it or move it beside the document"
                    )
                if isinstance(item, Mapping):
                    check_document(item, _path=f"{where}[{index}].")


def write_document(path: Path, document: Mapping[str, Any], force_new: bool = True) -> Path:
    """Write `document` as JSON, atomically, after checking it.

    The rename is atomic on POSIX, so a reader never sees a half-written
    document. That matters more than it looks: ``metadata.json`` is the
    idempotency cache key for the processing walk, and a truncated one would
    be read as "this frame is done".
    """
    check_document(document)

    if path.exists() and not force_new:
        raise FileExistsError(f"{path} exists and force_new is False")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp{os.getpid()}")
    try:
        temporary.write_text(json.dumps(document, indent=2, default=encode_value, sort_keys=True))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    return path


def read_document(path: Path) -> dict[str, Any] | None:
    """Read a metadata document, or return None if it is missing or unusable.

    A frame whose document cannot be read is treated exactly like one that has
    no document: it gets reprocessed. Raising here would mean a single bad file
    stops a walk over half a million frames, and the file is about to be
    rewritten anyway.

    "Unusable" includes a file that parses as valid JSON but is not an object.
    ``json.loads`` happily returns a list, a number or a string, and every
    caller here expects a mapping -- returning one would move the failure to an
    `AttributeError` at the call site, which is exactly the walk-stopping crash
    this function exists to prevent.
    """
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None

    return document if isinstance(document, dict) else None


def record_processing(
    metadata: dict[str, Any],
    params: PipelineParams,
    status: ImageStatus,
) -> dict[str, Any]:
    """Stamp the document with the settings and stage it was produced at.

    The fingerprint is what the next work-list walk compares against, so the
    field names live here rather than being spelled out at each call site.
    """
    metadata["image"]["params"] = json.loads(params.model_dump_json())
    metadata["image"]["params_fingerprint"] = params.fingerprint
    metadata["image"]["status"] = status.name
    return metadata


def write_frame(
    root: Path | str,
    path_info: ImagePathInfo,
    metadata: Mapping[str, Any],
    *,
    reduced: np.ndarray | None = None,
    extras: Mapping[str, np.ndarray] | None = None,
    sources: pandas.DataFrame | None = None,
    header: fits.Header | None = None,
    files: FileSettings | None = None,
    force_new: bool = True,
) -> dict[str, Path]:
    """Write one frame's document and products, returning what was written.

    Only `metadata` is required. The products are optional so a caller that
    has not plate-solved, or is re-emitting a document after a settings
    change, writes what it has instead of a file full of nulls. The returned
    mapping names only the files that were actually written.
    """
    files = files or FileSettings()
    directory = frame_directory(root, path_info)
    directory.mkdir(parents=True, exist_ok=True)

    written = {
        "metadata": write_document(
            directory / files.metadata_filename, metadata, force_new=force_new
        )
    }

    if reduced is not None:
        path = directory / files.reduced_filename
        fits.PrimaryHDU(reduced, header=header).writeto(path, overwrite=force_new)
        written["reduced"] = path

    if extras:
        path = directory / files.extras_filename
        hdul = fits.HDUList([fits.PrimaryHDU(header=header)])
        for name, data in extras.items():
            hdu = fits.ImageHDU(data, header=header)
            hdu.name = name.upper()
            hdul.append(hdu)
        hdul.writeto(path, overwrite=force_new)
        written["extras"] = path

    if sources is not None:
        path = directory / files.sources_filename
        sources.to_parquet(path)
        written["sources"] = path

    logger.debug(f"Wrote {len(written)} product(s) for {path_info.id} to {directory}")
    return written
