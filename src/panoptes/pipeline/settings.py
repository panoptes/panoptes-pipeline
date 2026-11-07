import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

#: Characters of the params digest kept as the fingerprint. Twelve hex
#: characters is 48 bits: collision-free across any plausible number of
#: parameter sets, and short enough to sit in a filename or a log line.
FINGERPRINT_LENGTH = 12


class CameraSettings(BaseModel):
    zero_bias: float = 512.0
    #: Raw ADU, as written in the file and as `WHTLVLN` reports it -- *not*
    #: post-bias. Saturation is a property of the detector, so it is measured
    #: and applied in the raw domain, before any bias is removed.
    #:
    #: This was `15872.0`, documented as "ADU after bias subtraction", which is
    #: exactly `2**14 - 512`: a 14-bit full scale with the bias already taken
    #: off. Feeding a raw `WHTLVLN` into that comparison would have masked
    #: roughly 512 ADU too aggressively. The value is unchanged in substance,
    #: only moved into the domain everything else uses.
    saturation: float = 16384.0
    effective_gain: float = 1.5
    image_width: int = 6000
    image_height: int = 4000
    #: Root of the camera registry, holding ``units/{unit_id}/cameras/{uid}.json``.
    #: None is the registry shipped with the package. Excluded from the dump and
    #: so from the fingerprint: where the registry sits changes no product, and
    #: what it says is fingerprinted per camera instead, in each frame's
    #: ``registry_fingerprint`` (data contract 6.2).
    registry_dir: Path | None = Field(default=None, exclude=True)


class BackgroundSettings(BaseModel):
    box_size: tuple[int, int] = (79, 84)
    filter_size: tuple[int, int] = (3, 3)


class StampSettings(BaseModel):
    """How the observation-level stamp cube is cut.

    Deliberately **not** part of `PipelineParams`, so it is not in the per-frame
    fingerprint: a stamp size changes the cube and nothing else, and folding it
    into the frame fingerprint would make every frame look stale -- a full
    re-solve of the sequence -- for a change that touches no frame. The cube
    records these settings, and the frames' fingerprint, in its own metadata.
    """

    #: Stamp width and height as a multiple of the sequence's median FWHM, in
    #: pixels, rounded up to a whole number of superpixels. The one algorithm
    #: parameter about the detector grid is that the result must be even, so
    #: every stamp carries the same Bayer phase; everything else is in FWHM
    #: units, per improvement plan 1.4.
    fwhm_multiple: float = 4.0
    #: Never cut a stamp smaller than this many pixels on a side, however sharp
    #: the sequence: a stamp has to hold a complete superpixel border around the
    #: star for the color masks to mean anything.
    min_size: int = 6
    #: A source must be matched in at least this fraction of the usable frames
    #: to get a stamp. Its stamp is then cut from *every* frame -- the position
    #: is a property of the sequence, and the pixels are there whether or not
    #: detection found the star that frame -- so this is a quality cut, not a
    #: completeness requirement. 1.0 is unusable on real data: over the 124
    #: frames of PAN012 2018-08-24 no star at all was matched in every frame,
    #: 9 were in 98% and 2,047 in 90%, because a handful of cloudy frames
    #: matched a few hundred stars instead of three thousand.
    min_frame_fraction: float = 0.9


class CatalogSettings(BaseModel):
    vmag_limits: tuple[float, float] = (6, 13)
    max_separation_arcsec: int = 25  # ~8-10 arcsec/pixel
    localbkg_width_pixels: int = 2
    detection_threshold: float = 10.0
    num_detect_pixels: int = 4
    #: Local PANOPTES Input Catalog: parquet, ECSV or CSV, by suffix. Required,
    #: as there is no network lookup -- `sources.get_stars` fails loudly if unset.
    catalog_filename: Path | None = None


class PipelineParams(BaseSettings):
    camera: CameraSettings = CameraSettings()
    catalog: CatalogSettings = CatalogSettings()
    background: BackgroundSettings = BackgroundSettings()

    @property
    def fingerprint(self) -> str:
        """A short digest of every parameter that affects a product.

        This is the cache key, not decoration. The old flow stored `params` in
        every document and never compared them, so changing a setting left
        stale products behind with no signal that they no longer matched the
        settings that were supposed to have produced them. See data contract
        3.4.

        Keys are sorted so the digest depends on the values and not on field
        declaration order -- otherwise reordering a model would invalidate the
        entire archive.
        """
        canonical = json.dumps(json.loads(self.model_dump_json()), sort_keys=True)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return digest[:FINGERPRINT_LENGTH]


class ObservationSettings(BaseModel):
    sequence_id: str
    process_images: bool = True
    upload: bool = True
    force_new: bool = False


class FileSettings(BaseModel):
    #: One multi-extension file per frame: PRIMARY is the reduced science
    #: frame, with BACKGROUND, RMS and MASK beside it. The reduced frame and
    #: its background model always change together -- `reduced = data -
    #: background` -- so splitting them across two files could only let them
    #: drift out of sync, which the work-list walk cannot detect because it
    #: reads only the metadata document.
    reduced_filename: Path = "image.fits"
    metadata_filename: Path = "metadata.json"
    sources_filename: Path = "sources.parquet"
    #: Observation-level products, written beside `observation.json`: the stamp
    #: cube and the positions it was cut at. Parquet, not HDF5 -- data contract
    #: 3.5. The two are separate files because they have different consumers:
    #: a benchmark reads both, a drift diagnostic reads only the positions.
    stamps_filename: Path = "stamps.parquet"
    positions_filename: Path = "positions.parquet"


class ImageSettings(BaseSettings):
    params: PipelineParams = PipelineParams()
    files: FileSettings = FileSettings()
    compress_fits: bool = True
    output_dir: Path
    upload: bool = True
    force_new: bool = False
