from typing import Union, Tuple

from pydantic import BaseSettings


class BackgroundParams(BaseSettings):
    camera_bias: Union[int, float] = 2048.
    box_size: Tuple[int, int] = (79, 84)
    filter_size: Tuple[int, int] = (11, 12)
    saturation: float = 13583.0  # ADU


class CatalogSearchParams(BaseSettings):
    vmag_min: float = 5
    vmag_max: float = 17
    numcont: int = 5
