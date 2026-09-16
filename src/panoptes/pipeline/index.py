"""Build the query surface by walking the documents the pipeline wrote.

`observations.csv` was a Firestore-generated summary. With no Firestore,
nothing produces the surface you query to ask "which observations do I have,
how long are they, how many frames are usable". This builds it from the
documents that are already on disk.

Two files, and the second is derived from the first:

``frames.parquet``
    One row per frame, from every ``metadata.json`` under the processed tree.

``observations.parquet``
    One row per sequence, from grouping ``frames.parquet``. **Never written
    directly from the tree.** If it were, the two could disagree and nothing
    would notice, which is the `observations.csv` failure one level up.

They are separate files rather than one table because they cannot share a
schema: ``image_calibration_saturation_provenance`` is meaningless for a
sequence and ``duration_minutes`` is meaningless for a frame. One table would
be half nulls and every query would have to filter on a row-type column to be
correct.

The rule that keeps this honest
-------------------------------
**The index must be deletable.** If it cannot be removed, rebuilt from the
tree, and give the same answers, something has entered the index that belongs
in the documents. That is not hypothetical: it is exactly how
`observations.csv` became the sole holder of `total_exptime`, which is now null
for every long sequence with nowhere to recover it from.

Nothing here opens a FITS file or reads a pixel. See data contract 4.2, and
#206 and #207 for the shape.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas
from loguru import logger

from panoptes.pipeline import products
from panoptes.pipeline.settings import FileSettings
from panoptes.pipeline.status import ImageStatus

#: Separator between levels of a flattened document key. Deliberately not `.`:
#: data contract 3.2 calls a dotted name a *view* over a nested map rather than
#: storage, and the dotted `camera.serial_number` in `observations.csv` is
#: exactly the confusion between "document path" and "column" this avoids.
SEPARATOR = "_"

#: Document blocks that do not become columns. `image.params` is a whole
#: settings dump per row, and `params_fingerprint` already summarizes it in one
#: column -- the fingerprint is the part with meaning, since it is what the
#: work-list walk compares.
DROPPED = (("image", "params"),)

FRAMES_FILENAME = "frames.parquet"
OBSERVATIONS_FILENAME = "observations.parquet"
SCHEMA_FILENAME = "schema.json"

#: Bumped when the column contract changes, so a reader can tell.
SCHEMA_VERSION = 1

#: Columns the observation index groups and aggregates on. Guaranteed present in
#: `frames.parquet` even when no document supplies them, because a tree of
#: documents written by an older pipeline must still index rather than raise --
#: and because a consumer needs one schema that does not depend on what happened
#: to be in the archive.
REQUIRED_FRAME_COLUMNS = (
    "unit_unit_id",
    "sequence_sequence_id",
    "sequence_sequence_time",
    "sequence_field_name",
    "sequence_camera_camera_id",
    "sequence_camera_serial_number",
    "image_uid",
    "image_image_time",
    "image_status",
    "image_camera_exptime",
    "image_params_fingerprint",
)

#: The observation index's columns, in order. Declared so an empty archive
#: writes the same schema as a full one.
OBSERVATION_COLUMNS = (
    "sequence_sequence_id",
    "unit_id",
    "camera_id",
    "field_name",
    "sequence_time",
    "num_frames",
    "num_usable",
    "start_time",
    "end_time",
    "duration_minutes",
    "total_exptime",
    "num_serials",
    "camera_num_serials",
)


def flatten(document: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested maps into single-level keys joined by `SEPARATOR`.

    ``{"image": {"camera": {"white_lvln": 11765}}}`` becomes
    ``{"image_camera_white_lvln": 11765}``. Lists are left alone: nothing in a
    metadata document holds one that wants to be a column, and `check_document`
    already refuses arrays of arrays.
    """
    flat: dict[str, Any] = {}
    for key, value in document.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}{SEPARATOR}"))
        else:
            flat[name] = value
    return flat


def find_documents(processed_root: Path | str, files: FileSettings | None = None) -> Iterator[Path]:
    """Every metadata document under `processed_root`, in a stable order.

    Sorted so two builds of the same tree produce rows in the same order, which
    is what lets the rebuild check compare frames directly.
    """
    files = files or FileSettings()
    return iter(sorted(Path(processed_root).rglob(str(files.metadata_filename))))


def build_frames(processed_root: Path | str, files: FileSettings | None = None) -> pandas.DataFrame:
    """One row per frame, from every document under `processed_root`.

    A document that cannot be read is skipped with a warning rather than
    stopping the walk -- over half a million frames, one bad file must not cost
    the whole index, and the frame it describes is still in the tree to be
    reprocessed.

    Missing fields become null columns rather than errors. Documents written by
    an older pipeline still have to index; that is the whole point of being able
    to rebuild.
    """
    rows = []
    for path in find_documents(processed_root, files):
        document = products.read_document(path)
        if document is None:
            logger.warning(f"Skipping unreadable document: {path}")
            continue

        for *parents, leaf in DROPPED:
            block = document
            for parent in parents:
                block = block.get(parent) if isinstance(block, dict) else None
            if isinstance(block, dict):
                block.pop(leaf, None)

        rows.append(flatten(document))

    # `pandas` unions the keys, so a field absent from one document arrives as a
    # null in that row rather than failing the build. `reindex` extends that to
    # a field absent from *every* document: without it, a tree written before a
    # field existed produces no column at all and the aggregation below raises
    # `KeyError` on exactly the old documents this is supposed to tolerate.
    frames = pandas.DataFrame(rows)
    missing = [c for c in REQUIRED_FRAME_COLUMNS if c not in frames.columns]
    if missing:
        logger.debug(f"No document supplied {missing}; indexing them as null.")
    return frames.reindex(columns=[*dict.fromkeys([*REQUIRED_FRAME_COLUMNS, *frames.columns])])


def build_observations(frames: pandas.DataFrame) -> pandas.DataFrame:
    """One row per sequence, grouped from `frames`.

    Four of these columns are the point, because the thing this replaces could
    not express them:

    ``num_usable``
        Frames that actually reached `ImageStatus.MATCHED`, as distinct from
        frames that exist. Answerable only because status is recorded per frame.

    ``total_exptime``
        A sum over frames rather than a number the index alone holds. That
        distinction is why the field is currently null for every long sequence
        in the archive with nowhere to recover it from. Summed with
        ``min_count=1``, so a sequence where *no* frame recorded an exposure
        stays null instead of reporting a confident zero -- the whole point
        being to stop missing data from looking like measured data.

    ``camera_num_serials``
        Distinct serials recorded against this camera uid **across every
        sequence in the index**. This is the defect data contract 2.3 describes:
        `14d3bd` carries 2,332 sequences on one serial and 4 on another. It is a
        property of the camera, not of a sequence, so grouping by sequence
        cannot see it -- within any one sequence the serial is almost always
        constant. Necessarily limited by what is indexed: an index covering one
        sequence cannot observe cross-sequence variation.

    ``num_serials``
        Distinct serials *within* one sequence. A different, rarer defect, kept
        because it is nearly free once the column is there.
    """
    if frames.empty:
        return pandas.DataFrame(columns=list(OBSERVATION_COLUMNS))

    table = frames.copy()
    table["image_time"] = pandas.to_datetime(
        table["image_image_time"], format="mixed", utc=True, errors="coerce"
    )

    # `groupby` drops null keys silently, which would make frames disappear from
    # the index with no signal. Say so instead.
    unplaceable = table["sequence_sequence_id"].isna().sum()
    if unplaceable:
        logger.warning(f"{unplaceable} frame(s) have no sequence id and are not indexed")
        table = table[table["sequence_sequence_id"].notna()]
    if table.empty:
        return pandas.DataFrame(columns=list(OBSERVATION_COLUMNS))

    # Serial cardinality per *camera*, across the whole table -- see the
    # docstring. Computed before grouping by sequence, because that grouping is
    # exactly what hides it.
    serials_per_camera = table.groupby("sequence_camera_camera_id", observed=True)[
        "sequence_camera_serial_number"
    ].nunique()

    grouped = table.groupby("sequence_sequence_id", observed=True)
    observations = grouped.agg(
        unit_id=("unit_unit_id", "first"),
        camera_id=("sequence_camera_camera_id", "first"),
        field_name=("sequence_field_name", "first"),
        sequence_time=("sequence_sequence_time", "first"),
        num_frames=("image_uid", "size"),
        start_time=("image_time", "min"),
        end_time=("image_time", "max"),
        total_exptime=("image_camera_exptime", lambda values: values.sum(min_count=1)),
        num_serials=("sequence_camera_serial_number", "nunique"),
    )
    observations["num_usable"] = grouped["image_status"].apply(
        lambda status: (status == ImageStatus.MATCHED.name).sum()
    )
    observations["duration_minutes"] = (
        observations.end_time - observations.start_time
    ).dt.total_seconds() / 60
    observations["camera_num_serials"] = observations.camera_id.map(serials_per_camera)

    inconsistent = sorted(serials_per_camera.index[serials_per_camera > 1].tolist())
    if inconsistent:
        logger.warning(
            f"Camera uid(s) {inconsistent} record more than one serial across sequences; "
            "this is a data defect rather than a camera swap -- see data contract 2.3"
        )

    return observations.reset_index()[list(OBSERVATION_COLUMNS)]


def schema(frames: pandas.DataFrame, observations: pandas.DataFrame) -> dict[str, Any]:
    """The column contract, as data rather than as source code.

    #207 requires the flattened-name mapping to be *recorded* where
    `panoptes-data` can read it. `SEPARATOR` and `DROPPED` are constants in this
    package, which a consumer in another repository cannot import and should not
    have to reverse-engineer from a parquet footer. This writes them down.

    No timestamps: a rebuild of an unchanged tree must produce an identical
    manifest, or the deletable rule stops being checkable.
    """
    return {
        "version": SCHEMA_VERSION,
        "separator": SEPARATOR,
        "dropped": [list(block) for block in DROPPED],
        "required_frame_columns": list(REQUIRED_FRAME_COLUMNS),
        "files": {
            FRAMES_FILENAME: {
                "row": "frame",
                "columns": {name: str(dtype) for name, dtype in frames.dtypes.items()},
            },
            OBSERVATIONS_FILENAME: {
                "row": "sequence",
                "derived_from": FRAMES_FILENAME,
                "columns": {name: str(dtype) for name, dtype in observations.dtypes.items()},
            },
        },
    }


def build(
    processed_root: Path | str,
    index_root: Path | str | None = None,
    files: FileSettings | None = None,
) -> dict[str, Path]:
    """Write both index files and the schema manifest, returning what was written.

    `index_root` defaults to the processed tree's own root, which keeps the
    index beside what it describes. It is regenerable, so nothing is lost by
    putting it somewhere else, or by deleting it.
    """
    index_root = Path(index_root if index_root is not None else processed_root)
    index_root.mkdir(parents=True, exist_ok=True)

    frames = build_frames(processed_root, files)
    observations = build_observations(frames)

    written = {
        "frames": index_root / FRAMES_FILENAME,
        "observations": index_root / OBSERVATIONS_FILENAME,
        "schema": index_root / SCHEMA_FILENAME,
    }
    frames.to_parquet(written["frames"], index=False)
    observations.to_parquet(written["observations"], index=False)
    written["schema"].write_text(json.dumps(schema(frames, observations), indent=2, sort_keys=True))

    logger.info(
        f"Indexed {len(frames)} frame(s) in {len(observations)} sequence(s) into {index_root}"
    )
    return written


def read(index_root: Path | str) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Read both index files back, frames first."""
    index_root = Path(index_root)
    return (
        pandas.read_parquet(index_root / FRAMES_FILENAME),
        pandas.read_parquet(index_root / OBSERVATIONS_FILENAME),
    )
