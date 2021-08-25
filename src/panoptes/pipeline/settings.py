from pathlib import Path
from typing import Tuple, Optional

from pydantic import BaseModel, BaseSettings


class CameraSettings(BaseModel):
    zero_bias: float = 2048.
    saturation: float = 11535.0  # ADU after bias subtraction.
    effective_gain: float = 1.5
    image_width: int = 6000
    image_height: int = 4000


class BackgroundSettings(BaseModel):
    box_size: Tuple[int, int] = (79, 84)
    filter_size: Tuple[int, int] = (3, 3)


class CatalogSettings(BaseModel):
    vmag_limits: Tuple[float, float] = (6, 14)
    numcont: int = 5
    max_separation_arcsec: int = 50
    localbkg_width_pixels: int = 2
    detection_threshold: float = 5.0
    num_detect_pixels: int = 4
    catalog_filename: Optional[Path] = None


class PipelineParams(BaseSettings):
    camera: CameraSettings = CameraSettings()
    catalog: CatalogSettings = CatalogSettings()
    background: BackgroundSettings = BackgroundSettings()
