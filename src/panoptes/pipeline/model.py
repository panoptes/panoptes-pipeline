from datetime import datetime
from typing import List, Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from pydantic import BaseModel, PositiveFloat, FilePath
from panoptes.utils.images import fits as fits_utils
from panoptes.pipeline.utils.metadata import extract_metadata


class Camera(BaseModel):
    id: int
    serial_number: int


class ImageData(BaseModel):
    data: Any = None
    background: Any = None
    background_rms: Any = None
    sources: Any = None
    wcs: Any = None
    header: Any = None


class ImageMetadata(BaseModel):
    unit_id: str
    camera_id: str
    sequence_id: str
    time: datetime
    exptime: PositiveFloat
    filepath: FilePath = None
    extra: dict = None


class Image(BaseModel):
    metadata: ImageMetadata = None
    data: ImageData = ImageData(),

    def save_fits(self, filename, **kwargs):
        hdu0 = fits.PrimaryHDU(self.data.data.data - self.data.background, header=self.data.header)
        hdu0.scale('float32')

        fits.HDUList(hdu0).writeto(filename, **kwargs)
        self.metadata.filepath = filename

    def __str__(self):
        return f'Image {self.metadata.filepath}'

    @classmethod
    def from_fits(cls, filename: FilePath):
        raw_data, header = fits_utils.getdata(str(filename), header=True)

        # Clear out bad headers.
        header['FILENAME'] = filename
        header.remove('COMMENT', ignore_missing=True, remove_all=True)
        header.remove('HISTORY', ignore_missing=True, remove_all=True)
        bad_headers = [h for h in header.keys() if h.startswith('_')]
        map(header.pop, bad_headers)

        metadata = extract_metadata(header)

        # Create a class instance.
        instance = cls(
            data=ImageData(data=raw_data.astype(np.float32)),
            metadata=ImageMetadata.parse_obj(metadata['image'])
        )
        instance.metadata.filepath = filename
        instance.data.header = header
        instance.data.wcs = WCS(header)

        # Add all metadata as extras.
        # TODO better parsing of the metadata.
        instance.metadata.extra = metadata

        return instance


class Observation(BaseModel):
    sequence_id: int
    images: List[Image]
    camera: Camera
    catalog_sources: Any
