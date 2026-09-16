"""The two processing entry points, and the per-frame work they consume.

The cloud flow had two abilities worth keeping exactly: process a single FITS
as it arrives, or process everything in a batch. Both survive here, with bucket
events replaced by directory enumeration and papermill replaced by ordinary
function calls. `process_frame` is the expensive per-FITS work; `process_
observation` runs the work-list walk, processes what needs it, and writes the
sequence document.

These are library functions, not commands. The CLI belongs to improvement plan
4.2 and should wrap these rather than contain them.

See data contract 5.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from loguru import logger
from panoptes.utils.images import bayer
from panoptes.utils.images import fits as fits_utils
from panoptes.utils.images.fits import ImagePathInfo

from panoptes.pipeline import products, provenance, worklist
from panoptes.pipeline.provenance import Resolved
from panoptes.pipeline.settings import FileSettings, ImageSettings, PipelineParams
from panoptes.pipeline.status import ImageStatus, ObservationStatus
from panoptes.pipeline.utils.images import (
    detect_sources,
    extract_metadata,
    match_sources,
    plate_solve,
)


class SolverMissing(RuntimeError):
    """`solve-field` is not installed, so a frame cannot be plate solved."""


@dataclass
class Calibrated:
    """One frame's pixels after bias, saturation masking and background.

    Both resolutions of the background are here because they serve different
    consumers: source detection needs it evaluated at the frame's shape, while
    only the mesh is worth writing to disk.
    """

    reduced: np.ma.MaskedArray
    background: np.ndarray
    rms: np.ndarray
    background_mesh: np.ndarray
    rms_mesh: np.ndarray
    mask: np.ndarray
    calibration: dict[str, Resolved]


def calibrate(raw_data: np.ndarray, header: fits.Header, params: PipelineParams) -> Calibrated:
    """Bias, saturation mask and RGB background, in that order.

    **Saturation is masked in the raw domain, before the bias is removed.**
    Saturation is a property of the detector, and `WHTLVLN` -- the only source
    that knows a given camera's real white level -- reports a raw ADU value.
    The old code masked after subtracting the bias against a threshold that was
    documented as post-bias (`15872`, which is `2**14 - 512`), so feeding a raw
    header value into that same comparison would have masked about 512 ADU too
    aggressively on every frame. Doing it first removes the ambiguity.

    The background is computed per Bayer colour, because the colours sit on
    different pedestals; that is why `get_rgb_background` exists rather than a
    single `Background2D`.
    """
    calibration = provenance.resolve_camera(header, params.camera)
    saturation = calibration["saturation"].value
    zero_bias = calibration["zero_bias"].value

    mask = np.asarray(raw_data) >= saturation
    if mask.any():
        logger.debug(f"Masked {mask.sum()} pixel(s) at or above {saturation} ADU")

    data = np.ma.array(np.asarray(raw_data, dtype="float32") - zero_bias, mask=mask)

    backgrounds = bayer.get_rgb_background(
        data=data,
        return_separate=True,
        box_size=params.background.box_size,
        filter_size=params.background.filter_size,
    )
    rgb_masks = bayer.get_rgb_masks(data)

    # Each colour's model covers the whole frame with the other colours masked,
    # so the full-resolution planes are summed back into one image while the
    # meshes are kept per colour -- the summation is what loses information, and
    # it is only needed for detection.
    background = np.ma.array(
        [np.ma.array(data=bg.background, mask=m) for bg, m in zip(backgrounds, rgb_masks)]
    )
    rms = np.ma.array(
        [np.ma.array(data=bg.background_rms, mask=m) for bg, m in zip(backgrounds, rgb_masks)]
    )
    background = background.filled(0).sum(0)
    rms = rms.filled(0).sum(0)

    return Calibrated(
        reduced=data - background,
        background=background,
        rms=rms,
        background_mesh=np.array([bg.background_mesh for bg in backgrounds]),
        rms_mesh=np.array([bg.background_rms_mesh for bg in backgrounds]),
        mask=mask,
        calibration=calibration,
    )


def require_solver() -> None:
    """Fail before any pixels are read if `solve-field` is not installed.

    Plate solving is not optional: a frame without a WCS cannot be matched to
    the catalogue, so it produces no photometry and there is nothing useful to
    write. Checking up front turns an hour of work that ends in an unusable
    tree into an immediate, actionable error.
    """
    if shutil.which("solve-field") is None:
        raise SolverMissing(
            "plate solving needs astrometry.net's `solve-field` on PATH; "
            "install it (e.g. `brew install astrometry-net` or "
            "`apt install astrometry.net`) along with index files covering "
            "the field scale"
        )


def process_frame(
    raw_path: Path | str,
    processed_root: Path | str,
    params: PipelineParams | None = None,
    *,
    force_new: bool = True,
    files: FileSettings | None = None,
    solve_timeout: int = 30,
) -> dict[str, Path]:
    """Process one raw frame and write its document and products.

    Bias, background, plate solve, source detection, catalogue match, then the
    artifacts. Plate solving belongs at this single-frame stage, which is why
    it is here rather than in the observation-level function.

    A frame that fails is still recorded, with `ImageStatus.ERROR`, so the next
    work-list walk reports `prior error` rather than `missing` and the failure
    is visible without re-reading logs.
    """
    raw_path = Path(raw_path)
    params = params or PipelineParams()
    files = files or FileSettings()

    require_solver()

    path_info = worklist.identify(raw_path)
    raw_data, header = fits_utils.getdata(str(raw_path), header=True)

    # The path is authoritative for identity; a header that disagrees with
    # where the file actually sits would put products in the wrong directory.
    header.setdefault("SEQID", path_info.sequence_id)
    header.setdefault("IMAGEID", path_info.image_id)

    directory = products.frame_directory(processed_root, path_info)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / files.reduced_filename

    try:
        calibrated = calibrate(raw_data, header, params)
        settings = ImageSettings(params=params, files=files, output_dir=directory)

        # Solve a scratch copy rather than the product. `solve-field` takes a
        # file, and even in `panoptes-utils`' non-destructive mode it leaves a
        # `.new` and a `.corr` beside whatever it is pointed at. Keeping that
        # in a temporary directory means the product is written exactly once,
        # already carrying its WCS, and no solver leftovers reach the archive.
        with tempfile.TemporaryDirectory() as scratch:
            scratch_path = Path(scratch) / "solve.fits"
            fits.PrimaryHDU(
                np.ma.filled(calibrated.reduced, np.nan).astype(np.float32), header=header
            ).writeto(scratch_path)
            wcs = plate_solve(settings=settings, filename=scratch_path, timeout=solve_timeout)

        header.update(wcs.to_header(relax=True))
        products.write_image(
            image_path,
            calibrated.reduced,
            background=calibrated.background_mesh,
            rms=calibrated.rms_mesh,
            mask=calibrated.mask,
            header=header,
            force_new=force_new,
        )

        detected = detect_sources(
            wcs, calibrated.reduced, calibrated.background, calibrated.rms, settings=settings
        )
        matched = match_sources(
            detected,
            wcs,
            settings=settings,
            image_width=calibrated.calibration["image_width"].value,
            image_height=calibrated.calibration["image_height"].value,
        )

        metadata = extract_metadata(header, path_info, params.camera)
        metadata["image"]["sources"] = source_statistics(matched)
        metadata = products.record_processing(metadata, params, ImageStatus.MATCHED)

        matched = matched.assign(uid=path_info.get_full_id())

    except Exception as error:
        logger.error(f"{path_info.id}: {error!r}")
        metadata = products.record_processing(
            extract_metadata(header, path_info, params.camera), params, ImageStatus.ERROR
        )
        metadata["image"]["error"] = repr(error)
        products.write_frame(processed_root, path_info, metadata, files=files, force_new=True)
        raise

    return products.write_frame(
        processed_root,
        path_info,
        metadata,
        sources=matched,
        files=files,
        force_new=force_new,
    )


def source_statistics(matched: pandas.DataFrame) -> dict[str, float | int]:
    """What the document records about a frame's sources.

    The FWHM is the useful one: it is what stamp sizes and apertures are
    expressed in multiples of, so it has to be recorded per frame rather than
    assumed.
    """
    fwhm_mean, fwhm_median, fwhm_std = sigma_clipped_stats(matched.photutils_fwhm)
    return dict(
        num_detected=len(matched),
        photutils_fwhm_mean=float(fwhm_mean),
        photutils_fwhm_median=float(fwhm_median),
        photutils_fwhm_std=float(fwhm_std),
    )


def process_observation(
    raw_root: Path | str,
    processed_root: Path | str,
    params: PipelineParams | None = None,
    *,
    force_new: bool = False,
    files: FileSettings | None = None,
    solve_timeout: int = 30,
) -> pandas.DataFrame:
    """Walk, process what needs processing, then write the sequence document.

    Returns the work list as a table with the outcome filled in, so the caller
    can see what was done and why without parsing logs. This is the deleted
    `observation.py` structure with `list_blobs` replaced by a glob.

    One frame's failure does not stop the batch: over a long sequence a single
    unsolvable frame is ordinary, and it is already recorded with
    `ImageStatus.ERROR` for the next walk to find.
    """
    params = params or PipelineParams()
    require_solver()

    frames = worklist.build(raw_root, processed_root, params, force_new=force_new, files=files)
    logger.info(f"Work list: {worklist.summarize(frames)}")

    outcomes = {}
    for frame in worklist.pending(frames):
        try:
            process_frame(
                frame.raw_path,
                processed_root,
                params,
                force_new=True,
                files=files,
                solve_timeout=solve_timeout,
            )
            outcomes[frame.path_info.image_id] = ImageStatus.MATCHED.name
        except Exception as error:
            logger.warning(f"{frame.path_info.image_id} failed: {error!r}")
            outcomes[frame.path_info.image_id] = ImageStatus.ERROR.name

    write_observation(processed_root, frames, params, files=files)

    table = worklist.as_table(frames)
    table["outcome"] = table.image_id.map(outcomes).fillna("skipped")
    return table


def write_observation(
    processed_root: Path | str,
    frames: list[worklist.Frame],
    params: PipelineParams,
    *,
    files: FileSettings | None = None,
) -> dict[str, Path]:
    """Aggregate a sequence's frames into one `observation.json` per sequence.

    This is the one file two frame workers would contend on, which is why
    nothing in `products.write_frame` touches it: the path hierarchy partitions
    every other write, so the per-frame work stays embarrassingly parallel with
    no shared mutable state. Here, at the end of a batch, there is exactly one
    writer.
    """
    files = files or FileSettings()
    written = {}

    for sequence_id, sequence_frames in group_by_sequence(frames).items():
        path_info = sequence_frames[0].path_info
        directory = products.frame_directory(processed_root, path_info).parent

        documents = [
            document
            for frame in sequence_frames
            if (
                document := products.read_document(
                    products.frame_directory(processed_root, frame.path_info)
                    / files.metadata_filename
                )
            )
            is not None
        ]
        processed = [d for d in documents if d.get("image", {}).get("status") == "MATCHED"]

        document = dict(
            sequence_id=sequence_id,
            unit_id=path_info.unit_id,
            camera_id=path_info.camera_id,
            sequence_time=path_info.sequence_time.to_datetime(),
            num_frames=len(sequence_frames),
            num_processed=len(processed),
            params_fingerprint=params.fingerprint,
            status=(
                ObservationStatus.MATCHED.name
                if processed and len(processed) == len(sequence_frames)
                else ObservationStatus.ERROR.name
                if not processed
                else ObservationStatus.PROCESSING.name
            ),
        )
        if processed:
            document["sequence"] = processed[0].get("sequence", {})

        written[sequence_id] = products.write_document(
            directory / "observation.json", document, force_new=True
        )

    return written


def group_by_sequence(frames: list[worklist.Frame]) -> dict[str, list[worklist.Frame]]:
    """Frames grouped by the sequence they belong to, order preserved."""
    grouped: dict[str, list[worklist.Frame]] = {}
    for frame in frames:
        grouped.setdefault(frame.path_info.sequence_id, []).append(frame)
    return grouped


def read_image(path: Path | str) -> dict[str, np.ndarray]:
    """Read back what `products.write_image` wrote, by extension name.

    ``PRIMARY`` comes back as ``reduced``. Present so a consumer does not have
    to know the extension order, only the names.
    """
    with fits.open(path) as hdul:
        planes = {"reduced": hdul[0].data}
        for hdu in hdul[1:]:
            if hdu.name:
                planes[hdu.name.lower()] = hdu.data
    return planes


def find_sequences(raw_root: Path | str) -> list[str]:
    """Every sequence id with at least one frame under `raw_root`."""
    frames = (worklist.identify(path) for path in worklist.find_frames(raw_root))
    return sorted({path_info.sequence_id for path_info in frames})


__all__ = [
    "Calibrated",
    "ImagePathInfo",
    "SolverMissing",
    "calibrate",
    "find_sequences",
    "group_by_sequence",
    "process_frame",
    "process_observation",
    "read_image",
    "require_solver",
    "source_statistics",
    "write_observation",
]
