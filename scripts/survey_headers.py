# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "astropy",
#   "pandas",
#   "panoptes-utils[images]>=0.3.1",
#   "pyarrow",
#   "tqdm",
# ]
# ///
"""Survey the FITS headers of a local copy of the *raw* archive.

**Reads headers, never pixels.** Nothing is calibrated, solved or written back;
`fits.open` is lazy, so each frame costs a few header blocks and is closed
again. This is a metadata pass over the archive, not a processing pass.

This is not `panoptes.pipeline.index`, and the difference is the point.

`panoptes.pipeline.index` builds the query surface from the documents the
pipeline wrote --
FWHM, source counts, plate-solved positions, all of them outputs of the
implementation being replaced. Selecting a benchmark dataset with those, in
order to judge new code, is circular and very hard to detect afterwards. It
also cannot run at all until a processed tree exists, which is the position we
are in now.

This walks raw frames instead and records what POCS wrote into the header,
before anything in this repository touched it. That is enough for:

- **The header vocabulary survey (#187).** Which keywords exist, in which POCS
  versions, on which camera families. A column's non-nullness across `creator`
  is the whole answer.
- **Benchmark selection, pass 1 (#95).** Frame count, duration, unit, camera
  uid, ISO, exposure, moon phase, airmass, field -- header facts every one, and
  the only cut that can honestly precede reprocessing.

Pass 2 -- drift and seeing measured on the pixels -- is deliberately *not* here.
It needs the frames reduced, so it happens after selection, on the candidates
this narrows to.

Layout
------
The archive keeps the bucket's layout, so a frame's path already carries its
identifiers::

    <root>/PAN012/358d0f/20180824T035917/20180824T040118.fits.fz
           unit     camera  sequence time  image time

Those are read from the path and cross-checked against ``SEQID``/``IMAGEID``
when the header carries them; a disagreement is recorded rather than resolved,
because which one is right is exactly the kind of thing this survey is for.

Output
------
One parquet shard per sequence under ``--output/shards``, then
``headers.parquet`` combining them. It is *not* called ``frames.parquet``:
`panoptes.pipeline.index` already writes a file by that name from the processed
documents, and the two answer different questions from different sources.

Sharding is what makes a run over ten years resumable: a shard that exists is
skipped, so an interrupted run continues rather than restarts.

A frame that cannot be read gets a row with ``error`` set instead of raising.
Over an archive this size a crash on frame 40,000 loses the run, and a header
that fails to parse is itself a finding. The count is reported at the end.

Usage
-----
::

    uv run scripts/survey_headers.py /data/panoptes-archive
    uv run scripts/survey_headers.py /data/panoptes-archive --jobs 16
    uv run scripts/survey_headers.py /data/panoptes-archive --units PAN012 PAN007
"""

from __future__ import annotations

import argparse
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from panoptes.utils.images import fits as fits_utils
from panoptes.utils.images.fits import ImagePathInfo
from tqdm import tqdm

#: Cards that say nothing about the observation under any option. The `Z*`
#: block describes the fpack tiling, `TTYPE`/`TFORM` the compressed table, and
#: the `_`-prefixed cards are astrometry.net's *truncated* duplicates of its own
#: WCS -- malformed keywords like `_RVAL1` and `__ORDER`, never worth keeping.
NOISE_PREFIXES = ("_", "Z", "TTYPE", "TFORM")

#: The SIP distortion blocks. Solver output like `SOLVER_KEYS`, and kept by the
#: same `--keep-wcs`: excluding them there would hand back a WCS that cannot
#: actually be applied, since the distortion is most of what it is worth.
WCS_PREFIXES = ("A_", "B_", "AP_", "BP_", "PV")
WCS_KEYS = frozenset({"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"})
#: Cards written by astrometry.net, not by POCS. The old cloud pipeline solved
#: frames and wrote the WCS *back into the archived file*, so a 2018 frame is
#: not the raw header POCS produced. Keeping these as if they were header facts
#: is what makes benchmark selection circular -- they are output of the
#: implementation being replaced. `ra_mnt`/`dec_mnt` are the mount's own
#: readings and are the honest pointing. `--keep-wcs` overrides.
SOLVER_KEYS = frozenset(
    {
        "CTYPE1",
        "CTYPE2",
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CUNIT1",
        "CUNIT2",
        "CDELT1",
        "CDELT2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
        "WCSAXES",
        "RADESYS",
        "RADECSYS",
        "EQUINOX",
        "LONPOLE",
        "LATPOLE",
        "IMAGEW",
        "IMAGEH",
        "STATUS",
    }
)

NOISE_KEYS = frozenset(
    {
        "",
        "COMMENT",
        "HISTORY",
        "CHECKSUM",
        "DATASUM",
        "SIMPLE",
        "EXTEND",
        "XTENSION",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTNAME",
        "BSCALE",
        "BZERO",
    }
)

#: A sequence directory is named for its start time, at either depth: the
#: modern layout is `<unit>/<camera>/<sequence time>/`, and the legacy one
#: carries a field name as well, `<unit>/<field>/<camera>/<sequence time>/`.
SEQUENCE_DIR = re.compile(r"^\d{8}T\d{6}$")


SOLVE_HISTORY = re.compile(r"Plate-solved by .* at (?P<when>[\d\-: ]+)")


def solve_provenance(header: fits.Header) -> dict:
    """Whether this archived frame was written back to by a solver, and when.

    This is the fact worth keeping. The WCS itself is a measurement made by the
    code under replacement; *that a solve happened* is provenance, and it varies
    by era -- which is a #187 finding in its own right, since it says the
    archive's "raw" frames are raw to different degrees.
    """
    solved = "CRVAL1" in header or str(header.get("STATUS", "")).strip() == "solved"
    row: dict = {"plate_solved": solved}
    for card in header.get("HISTORY", []):
        match = SOLVE_HISTORY.search(str(card))
        if match:
            row["solve_date"] = match.group("when").strip()
            break
    return row


def is_noise(key: str, *, keep_wcs: bool = False) -> bool:
    """Whether a header keyword carries nothing about the observation.

    `keep_wcs` exempts the solver's WCS and SIP cards, which are excluded by
    default as circular for benchmark selection but are the whole point of the
    option when it is set.
    """
    if keep_wcs and (key in WCS_KEYS or key in SOLVER_KEYS or key.startswith(WCS_PREFIXES)):
        return False
    if key in WCS_KEYS or key.startswith(WCS_PREFIXES):
        return True
    if key in NOISE_KEYS:
        return True
    # `Z` alone would eat `ZP`-style keys, so require the compression block's
    # actual shape: `ZIMAGE`, `ZNAXIS1`, `ZVAL2`, `ZHECKSUM`.
    if key.startswith("Z") and key not in {"ZP", "ZPT"}:
        return True
    return key.startswith(NOISE_PREFIXES)


def column_name(key: str) -> str:
    """`RA-MNT` -> `ra_mnt`. Header keywords are not valid pandas names."""
    return key.lower().replace("-", "_")


def read_frame(path: Path, root: Path, *, keep_wcs: bool = False) -> dict:
    """One row: identifiers from the path, everything else from the header."""
    relative = path.relative_to(root).as_posix()
    row: dict = {"path": relative}

    # `ImagePathInfo` is the repository's shared reading of an archive path. It
    # also understands the legacy `<unit>/<field>/<camera>/...` layout, which a
    # regex written against the modern one drops without saying so.
    try:
        info = ImagePathInfo(path=relative)
        row |= {
            "unit_id": info.unit_id,
            "camera_id": info.camera_id,
            "sequence_id": info.sequence_id,
            "image_id": info.image_id,
            # `ImagePathInfo` gives astropy `Time`; parquet wants a datetime,
            # and `image_time` is what a sequence's duration is measured from.
            "sequence_time": info.sequence_time.datetime,
            "image_time": info.image_time.datetime,
        }
    except Exception as exc:  # noqa: BLE001 -- an unparseable path is a finding
        row["error"] = f"path: {type(exc).__name__}: {exc}"
        return row

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            # Picks the compressed image extension for `.fz`, ext 0 otherwise.
            header = fits_utils.getheader(path)
    except Exception as exc:  # noqa: BLE001 -- an unreadable frame is a finding
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row |= solve_provenance(header)
    row["n_cards"] = len(header)
    row["file_bytes"] = path.stat().st_size
    # `ZNAXIS*` is the shape of the image inside the compression; `NAXIS*` is
    # the shape of the table holding it, which is not the frame.
    row["naxis1"] = header.get("ZNAXIS1", header.get("NAXIS1"))
    row["naxis2"] = header.get("ZNAXIS2", header.get("NAXIS2"))

    for key in header:
        if is_noise(key, keep_wcs=keep_wcs) or (key in SOLVER_KEYS and not keep_wcs):
            continue
        value = header[key]
        if isinstance(value, fits.card.Undefined):
            continue
        row.setdefault(column_name(key), value)

    # The path and the header are two independent claims about identity. Record
    # a disagreement; do not pick a winner. Both are checked, because they fail
    # apart: `SEQID` catches a frame filed under the wrong sequence, `IMAGEID`
    # catches one whose own timestamp does not match its filename.
    seqid = header.get("SEQID")
    if seqid and seqid != row["sequence_id"]:
        row["seqid_disagrees"] = True
    imageid = header.get("IMAGEID")
    if imageid and imageid != row["image_id"]:
        row["imageid_disagrees"] = True

    return row


def as_text(value: object) -> object:
    """Render one header value as text without inventing digits.

    A camera serial arrives as the string ``'012070048413'`` from one body and
    as the float ``273074013298.0`` from another. `str()` on the second yields
    a trailing ``.0`` that no header contains, so an integral float is rendered
    as an integer. The *leading zero* difference survives, because that one is
    real and is what #187 is looking for.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, bool | int | str):
        return str(value)
    return str(value)


def harmonize(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every column one type, without inventing or destroying values.

    The fleet writes the same keyword with different types. `CAMSN` is the
    string ``'022071154939'`` on one body and a bare number on another, and a
    column holding both cannot be written to parquet at all.

    The rule is **one string poisons the column**: if any frame wrote this
    keyword as text, every value becomes text. Not the other way around --
    coercing toward numbers would read ``'022071154939'`` as 22071154939 and
    silently drop a leading zero from a camera serial, which is the one column
    the camera registry is keyed on.

    Idempotent, so applying it per shard and again over their concatenation
    gives the same answer as applying it once to everything.
    """
    for name, column in frame.items():
        if column.dtype != object:
            continue
        present = column.dropna()
        if present.empty:
            continue
        if present.map(lambda v: isinstance(v, str)).any():
            frame[name] = column.where(column.isna(), column.map(as_text))
        else:
            # Numbers and booleans only. `infer_objects` leaves anything it
            # cannot type alone, which is the behavior wanted here.
            frame[name] = column.infer_objects()
    return frame


def shard_name(sequence_dir: Path, root: Path) -> str:
    """The sequence directory's path under the archive, flattened.

    Derived from the path rather than from a parsed sequence id because the two
    archive layouts put the camera at different depths, and a shard has to be
    named before any file in it has been read.
    """
    return sequence_dir.relative_to(root).as_posix().replace("/", "-")


def find_sequence_dirs(unit_dir: Path) -> list[Path]:
    """Every sequence directory under one unit, in either archive layout.

    Globbed at the two known depths rather than walked recursively: on a
    networked copy of the archive a full `rglob` over ten years of frames costs
    far more than two bounded globs.
    """
    candidates = [*unit_dir.glob("*/*"), *unit_dir.glob("*/*/*")]
    return sorted(d for d in candidates if d.is_dir() and SEQUENCE_DIR.match(d.name))


def index_sequence(args: tuple[Path, Path, Path, bool]) -> tuple[str, int, int]:
    """Index one sequence directory into its own shard.

    The shard is per *sequence* rather than per unit because units differ in
    size by orders of magnitude: one unit holding a third of the archive would
    leave a single worker running long after the pool went idle, and an
    interrupted run would lose all of it. Returns (name, frames, errors).
    """
    sequence_dir, root, output, keep_wcs = args
    shard = output / f"{shard_name(sequence_dir, root)}.parquet"

    rows = [read_frame(p, root, keep_wcs=keep_wcs) for p in sorted(sequence_dir.glob("*.fits*"))]
    if not rows:
        return shard.stem, 0, 0

    # A column entirely absent here is meaningful -- that sequence's POCS never
    # wrote it -- so keep the sparse shape rather than dropping empty columns.
    frame = harmonize(pd.DataFrame(rows))
    frame.to_parquet(shard, index=False)
    errors = int(frame["error"].notna().sum()) if "error" in frame else 0
    return shard.stem, len(frame), errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="directory holding the PANnnn unit folders")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("header-survey"),
        help="where the shards and headers.parquet go (default: ./header-survey)",
    )
    parser.add_argument("--units", nargs="*", help="only these units (default: all)")
    parser.add_argument("--jobs", type=int, default=8, help="parallel workers (default: 8)")
    parser.add_argument(
        "--rebuild", action="store_true", help="re-index sequences whose shard already exists"
    )
    parser.add_argument(
        "--keep-wcs",
        action="store_true",
        help="also record the WCS the old cloud pipeline wrote back into archived frames "
        "(solver output, not POCS header facts -- circular for benchmark selection)",
    )
    args = parser.parse_args()

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    output: Path = args.output.expanduser().resolve()
    # The shard directory is named for the options that produced its contents.
    # Keying reuse on the sequence path alone would let a `--keep-wcs` rerun
    # skip shards extracted without it and report a result that silently lacks
    # the WCS that was asked for -- and the reverse.
    shard_dir = output / ("shards-keep-wcs" if args.keep_wcs else "shards")
    shard_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.units) if args.units else None
    units = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith("PAN") and (wanted is None or d.name in wanted)
    )
    if not units:
        raise SystemExit(f"no PANnnn unit directories under {root}")

    print(f"{len(units)} unit(s); scanning for sequences...")
    sequences = sorted(seq for unit in units for seq in find_sequence_dirs(unit))
    print(f"{len(sequences)} sequence(s)")

    payload = [
        (seq, root, shard_dir, args.keep_wcs)
        for seq in sequences
        if args.rebuild or not (shard_dir / f"{shard_name(seq, root)}.parquet").exists()
    ]
    skipped = len(sequences) - len(payload)
    if skipped:
        print(f"{skipped} already indexed; --rebuild to redo them")

    if payload:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for _ in tqdm(
                pool.map(index_sequence, payload, chunksize=4),
                total=len(payload),
                desc="sequences",
            ):
                pass

    # Built from the sequences this run asked for, never from a glob over the
    # directory: after a full survey, `--units PAN012` would otherwise combine
    # every unit already on disk and write a `headers.parquet` far wider than
    # the scope requested. Resuming still works, because `sequences` is the
    # whole requested scope while `payload` was only the part not yet done.
    shards = [shard_dir / f"{shard_name(seq, root)}.parquet" for seq in sequences]
    missing = [p for p in shards if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} shard(s) missing, first: {missing[0]}")
    if not shards:
        raise SystemExit("no shards were written")

    print(f"combining {len(shards)} shard(s)...")
    combined = harmonize(pd.concat((pd.read_parquet(s) for s in shards), ignore_index=True))
    combined_path = output / "headers.parquet"
    combined.to_parquet(combined_path, index=False)

    print(f"\n{len(combined)} frames, {len(combined.columns)} columns -> {combined_path}")
    print(f"{combined_path.stat().st_size / 1e6:.1f} MB")
    # Counted from the combined table rather than from this run, which would
    # report zero for a resumed survey whose errors are all in earlier shards.
    errors = int(combined["error"].notna().sum()) if "error" in combined else 0
    if errors:
        print(f"{errors} frame(s) recorded with an error")
    if "plate_solved" in combined:
        solved = int(combined["plate_solved"].fillna(False).sum())
        print(f"{solved} of {len(combined)} frames were written back to by a solver")
    if "creator" in combined:
        print("\nframes per POCS version:")
        print(combined["creator"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
