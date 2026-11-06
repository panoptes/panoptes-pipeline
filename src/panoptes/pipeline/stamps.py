"""The observation-level product: one stamp cube per sequence, cut from reduced frames.

`process_frame` leaves a sequence as a tree of per-frame products -- a reduced
`image.fits` and a matched `sources.parquet` each. The algorithm wants the
other shape: for every star, its pixels in every frame. This module turns the
first into the second.

Stamps are cut at one position per star for the whole sequence -- the median of
its matched catalog position -- snapped to the superpixel grid so every stamp
carries the same Bayer phase, and sized as a multiple of the sequence's median
FWHM. Per-frame re-centering is the first thing M2 (drift) changes about this,
deliberately not done here: #94 wants a substrate that is clean, not one that
is already tuned.

What is written beside `observation.json`, as parquet (data contract 3.5):

- `stamps.parquet` -- indexed on ``(picid, time)``, one ``pixel_NNN`` column
  per pixel, row-major so the flat axis unpacks as ``(height, width)``.
- `positions.parquet` -- indexed on ``picid``: where each stamp was cut and the
  position statistics it was cut from. Its parquet schema metadata carries
  ``stamp_width``, ``stamp_height`` and how the cube was built.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas
import pyarrow
import pyarrow.parquet as parquet
from astropy.io import fits
from loguru import logger
from panoptes.utils.images import bayer

from panoptes.pipeline import products
from panoptes.pipeline.settings import FileSettings, StampSettings
from panoptes.pipeline.status import ImageStatus
from panoptes.pipeline.utils.observations import make_stamps

#: Parquet schema metadata key under which `positions.parquet` records how the
#: cube was built. The same convention `scripts/fetch_catalog.py` uses for the
#: catalog's query.
METADATA_KEY = b"panoptes_stamps"

#: The per-frame column pair the stamps are centered on. The catalog position
#: projected through the frame's WCS, rather than the detected centroid,
#: because the centroid of a star on a Bayer array is biased by which color
#: its core lands on -- which is the systematic being measured, not a
#: coordinate to center on.
POSITION_COLUMNS = ("catalog_wcs_x", "catalog_wcs_y")


class NoUsableFrames(RuntimeError):
    """No frame in the sequence reached `ImageStatus.MATCHED`."""


class NoCompleteSources(RuntimeError):
    """No source was matched in enough frames to get a stamp."""


@dataclass
class UsableFrame:
    """One frame that reached `MATCHED`, and where its products are."""

    directory: Path
    time: pandas.Timestamp
    fwhm: float
    image_width: int
    image_height: int
    params_fingerprint: str


def usable_frames(sequence_dir: Path | str, files: FileSettings | None = None) -> list[UsableFrame]:
    """The frames under `sequence_dir` whose document says they matched, by time.

    A frame recorded as `ERROR` is skipped silently here: it was already logged
    when it failed, and the next work-list walk will offer it again. A document
    that cannot be read is skipped with a warning, same as the index does.
    """
    files = files or FileSettings()
    frames = []
    for path in sorted(Path(sequence_dir).glob(f"*/{files.metadata_filename}")):
        document = products.read_document(path)
        if document is None:
            logger.warning(f"Skipping unreadable document: {path}")
            continue
        image = document.get("image", {})
        if image.get("status") != ImageStatus.MATCHED.name:
            continue
        calibration = image.get("calibration", {})
        frames.append(
            UsableFrame(
                directory=path.parent,
                time=pandas.Timestamp(image["image_time"]),
                fwhm=float(image["sources"]["photutils_fwhm_median"]),
                image_width=int(calibration["image_width"]["value"]),
                image_height=int(calibration["image_height"]["value"]),
                params_fingerprint=str(image.get("params_fingerprint", "")),
            )
        )
    return sorted(frames, key=lambda frame: frame.time)


def collect_sources(frames: list[UsableFrame], files: FileSettings | None = None) -> pandas.DataFrame:
    """Every frame's matched sources in one table, indexed on ``(picid, time)``."""
    files = files or FileSettings()
    tables = []
    for frame in frames:
        table = pandas.read_parquet(frame.directory / files.sources_filename)
        table["time"] = frame.time
        tables.append(table)
    sources = pandas.concat(tables, ignore_index=True)
    return sources.set_index(["picid", "time"]).sort_index()


def stamp_size(fwhm: float, settings: StampSettings) -> int:
    """Stamp side in pixels: a multiple of the FWHM, rounded up to whole superpixels.

    Even by construction, so `bayer.get_stamp_slice` can place a complete
    superpixel border around the star and every stamp in the cube carries the
    same Bayer phase. See improvement plan 1.4 for why the only pixel unit
    here is the detector grid.
    """
    size = max(math.ceil(settings.fwhm_multiple * fwhm), settings.min_size)
    return size + (size % 2)


def complete_sources(sources: pandas.DataFrame, num_frames: int, settings: StampSettings) -> pandas.Index:
    """The `picid`s matched in at least `min_frame_fraction` of `num_frames`."""
    counts = sources.groupby(level="picid", observed=True).size()
    needed = math.ceil(settings.min_frame_fraction * num_frames)
    return counts.index[counts >= needed]


def stamp_positions(
    sources: pandas.DataFrame,
    picids: pandas.Index,
    size: int,
    image_width: int,
    image_height: int,
) -> pandas.DataFrame:
    """Where each star's stamp is cut, for the whole sequence.

    The median matched position over the sequence, through
    `bayer.get_stamp_slice` so the origin lands on a superpixel boundary.
    Stars whose stamp would cross the frame edge are dropped here rather than
    at extraction, so `positions.parquet` describes exactly what is in the cube.
    """
    x_column, y_column = POSITION_COLUMNS
    selected = sources.loc[sources.index.get_level_values("picid").isin(picids), [x_column, y_column]]
    grouped = selected.groupby(level="picid", observed=True)
    center = grouped.median()
    spread = grouped.std().fillna(0.0)

    slices = np.array(
        [
            bayer.get_stamp_slice(float(x), float(y), stamp_size=(size, size), as_slices=False)
            for x, y in zip(center[x_column], center[y_column])
        ]
    ).reshape(-1, 4)

    positions = pandas.DataFrame(
        {
            f"{x_column}_median": center[x_column].to_numpy(),
            f"{y_column}_median": center[y_column].to_numpy(),
            f"{x_column}_std": spread[x_column].to_numpy(),
            f"{y_column}_std": spread[y_column].to_numpy(),
            "stamp_y_min": slices[:, 0],
            "stamp_y_max": slices[:, 1],
            "stamp_x_min": slices[:, 2],
            "stamp_x_max": slices[:, 3],
        },
        index=center.index,
    )

    inside = (
        (positions.stamp_x_min >= 0)
        & (positions.stamp_y_min >= 0)
        & (positions.stamp_x_max <= image_width)
        & (positions.stamp_y_max <= image_height)
    )
    if not inside.all():
        logger.info(f"Dropping {int((~inside).sum())} source(s) whose stamp would cross the frame edge")
    return positions.loc[inside]


def read_reduced(path: Path | str) -> np.ndarray:
    """The reduced science frame, as written by `products.write_image`.

    ``PRIMARY`` only; masked pixels come back as the NaN they were written as.
    """
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)


def extract(
    frames: list[UsableFrame],
    positions: pandas.DataFrame,
    files: FileSettings | None = None,
) -> pandas.DataFrame:
    """Cut every stamp from every frame into one table indexed on ``(picid, time)``."""
    files = files or FileSettings()
    tables = []
    for frame in frames:
        reduced = read_reduced(frame.directory / files.reduced_filename)
        stamps = make_stamps(positions, reduced)
        stamps.index = pandas.MultiIndex.from_arrays(
            [stamps.index, np.repeat(frame.time, len(stamps))], names=("picid", "time")
        )
        tables.append(stamps)
    return pandas.concat(tables).sort_index()


def drop_masked(stamps: pandas.DataFrame) -> tuple[pandas.DataFrame, list[int]]:
    """Remove every source with a masked pixel in any frame.

    The reduced frame carries saturation as NaN, and a NaN anywhere in a star's
    cube poisons every sum taken over it. These are the saturated stars, which
    no aperture photometry recovers; dropping them whole is honest where
    interpolating the pixel would not be. The caller records how many went.
    """
    pixel_columns = [c for c in stamps.columns if c.startswith("pixel_")]
    poisoned = stamps[pixel_columns].isna().any(axis=1).groupby(level="picid", observed=True).any()
    dropped = poisoned.index[poisoned].tolist()
    if dropped:
        logger.info(f"Dropping {len(dropped)} source(s) with a masked pixel in at least one frame")
        stamps = stamps.loc[~stamps.index.get_level_values("picid").isin(dropped)]
    return stamps, dropped


def write(
    sequence_dir: Path | str,
    stamps: pandas.DataFrame,
    positions: pandas.DataFrame,
    metadata: dict[str, Any],
    files: FileSettings | None = None,
    force_new: bool = True,
) -> dict[str, Path]:
    """Write the cube and its positions beside `observation.json`."""
    files = files or FileSettings()
    sequence_dir = Path(sequence_dir)
    stamps_path = sequence_dir / files.stamps_filename
    positions_path = sequence_dir / files.positions_filename
    for path in (stamps_path, positions_path):
        if path.exists() and not force_new:
            raise FileExistsError(f"{path} exists and force_new is False")

    stamps.to_parquet(stamps_path)

    table = pyarrow.Table.from_pandas(positions)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), METADATA_KEY: json.dumps(metadata, default=str).encode()}
    )
    parquet.write_table(table, positions_path)
    return {"stamps": stamps_path, "positions": positions_path}


def read(
    sequence_dir: Path | str, files: FileSettings | None = None
) -> tuple[pandas.DataFrame, pandas.DataFrame, dict]:
    """Read back what `write` wrote: ``(stamps, positions, metadata)``."""
    files = files or FileSettings()
    sequence_dir = Path(sequence_dir)
    stamps = pandas.read_parquet(sequence_dir / files.stamps_filename)
    table = parquet.read_table(sequence_dir / files.positions_filename)
    metadata = json.loads((table.schema.metadata or {}).get(METADATA_KEY, b"{}"))
    return stamps, table.to_pandas(), metadata


class MixedFingerprints(RuntimeError):
    """The usable frames were not all produced with the same params."""


def build(
    sequence_dir: Path | str,
    settings: StampSettings | None = None,
    *,
    files: FileSettings | None = None,
    force_new: bool = True,
) -> dict[str, Path]:
    """Cut the stamp cube for one processed sequence and write it.

    `sequence_dir` is the directory holding `observation.json` and one
    subdirectory per frame, as `process_observation` leaves it. Every usable
    frame has to carry the same params fingerprint: a cube mixing frames
    reduced under two settings would be a product of neither.
    """
    settings = settings or StampSettings()
    files = files or FileSettings()
    sequence_dir = Path(sequence_dir)

    frames = usable_frames(sequence_dir, files)
    if not frames:
        raise NoUsableFrames(f"no frame under {sequence_dir} reached {ImageStatus.MATCHED.name}")

    fingerprints = sorted({frame.params_fingerprint for frame in frames})
    if len(fingerprints) != 1:
        raise MixedFingerprints(
            f"frames under {sequence_dir} carry {len(fingerprints)} fingerprints: {fingerprints}"
        )

    sources = collect_sources(frames, files)
    picids = complete_sources(sources, len(frames), settings)
    if len(picids) == 0:
        raise NoCompleteSources(
            f"no source was matched in {settings.min_frame_fraction:.0%} of {len(frames)} frames"
        )

    fwhm = float(np.median([frame.fwhm for frame in frames]))
    size = stamp_size(fwhm, settings)
    positions = stamp_positions(sources, picids, size, frames[0].image_width, frames[0].image_height)
    if len(positions) == 0:
        raise NoCompleteSources(
            f"every one of the {len(picids)} complete sources has a {size}x{size} stamp crossing the "
            "frame edge; a cube with no positions is not a product"
        )
    logger.info(
        f"{sequence_dir.name}: {len(positions)} sources x {len(frames)} frames, "
        f"{size}x{size} stamps from a median FWHM of {fwhm:.2f} px"
    )

    stamps = extract(frames, positions, files)
    stamps, masked = drop_masked(stamps)
    positions = positions.loc[positions.index.isin(stamps.index.get_level_values("picid").unique())]

    metadata = dict(
        stamp_width=size,
        stamp_height=size,
        fwhm_median=fwhm,
        num_frames=len(frames),
        num_sources=len(positions),
        num_candidates=int(len(picids)),
        num_masked_dropped=len(masked),
        position_columns=list(POSITION_COLUMNS),
        frames_fingerprint=fingerprints[0],
        stamps=settings.model_dump(),
    )
    return write(sequence_dir, stamps, positions, metadata, files=files, force_new=force_new)


__all__ = [
    "METADATA_KEY",
    "MixedFingerprints",
    "NoCompleteSources",
    "NoUsableFrames",
    "UsableFrame",
    "build",
    "collect_sources",
    "complete_sources",
    "drop_masked",
    "extract",
    "read",
    "read_reduced",
    "stamp_positions",
    "stamp_size",
    "usable_frames",
    "write",
]
