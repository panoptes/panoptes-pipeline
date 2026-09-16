import hashlib
import json
from pathlib import Path

from pydantic import BaseModel
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


class BackgroundSettings(BaseModel):
    box_size: tuple[int, int] = (79, 84)
    filter_size: tuple[int, int] = (3, 3)


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


class ImageSettings(BaseSettings):
    params: PipelineParams = PipelineParams()
    files: FileSettings = FileSettings()
    compress_fits: bool = True
    output_dir: Path
    upload: bool = True
    force_new: bool = False
