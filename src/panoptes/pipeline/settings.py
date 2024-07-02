from pathlib import Path
from typing import Tuple, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class CameraSettings(BaseModel):
    zero_bias: float = 2048.
    saturation: float = 11530.0  # ADU after bias subtraction.
    effective_gain: float = 1.5
    image_width: int = 6000
    image_height: int = 4000


class BackgroundSettings(BaseModel):
    box_size: Tuple[int, int] = (79, 84)
    filter_size: Tuple[int, int] = (3, 3)


class CatalogSettings(BaseModel):
    vmag_limits: Tuple[float, float] = (6, 13)
    max_separation_arcsec: int = 25  # ~8-10 arcsec/pixel
    localbkg_width_pixels: int = 2
    detection_threshold: float = 10.0
    num_detect_pixels: int = 4
    catalog_filename: Optional[Path] = None


class PipelineParams(BaseSettings):
    camera: CameraSettings = CameraSettings()
    catalog: CatalogSettings = CatalogSettings()
    background: BackgroundSettings = BackgroundSettings()


class ObservationSettings(BaseModel):
    sequence_id: str
    process_images: bool = True
    upload: bool = True
    force_new: bool = False


class FileSettings(BaseModel):
    reduced_filename: Path = 'image.fits'
    extras_filename: Path = 'extras.fits'
    metadata_filename: Path = 'metadata.json'
    sources_filename: Path = 'sources.parquet'


class ImageSettings(BaseSettings):
    params: PipelineParams = PipelineParams()
    files: FileSettings = FileSettings()
    compress_fits: bool = True
    output_dir: Path
