"""
Example to read a FITS file.

Created on Jul 9, 2019

Be aware that hdus.close () needs to be called to limit the number of open files at a given time.

@author: skwok
"""

import warnings
import numpy as np

import astropy.io.fits as pf
from astropy.utils.exceptions import AstropyWarning

from keckdrpframework.models.arguments import Arguments
from keckdrpframework.primitives.base_primitive import BasePrimitive


def open_nowarning(filename):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        return pf.open(filename, memmap=False)


class ReadFits(BasePrimitive):
    def __init__(self, action, context):
        """
        Initializes the super class.
        """
        BasePrimitive.__init__(self, action, context)

    def _perform(self):
        """
        Expects action.args.name as fits file name
        Returns HDUs or (later) data model
        """
        name = self.action.args.name
        self.logger.debug(f"Reading {name}")
        out_args = Arguments()
        out_args.name = name
        out_args.image_data = self.readData(name)

        return out_args

    def _split_size(self, desc):
        if desc is None:
            return None
        p1, p2 = desc[1:-1].split(",")
        x0, x1 = p1.split(":")
        y0, y1 = p2.split(":")
        return [int(x) for x in (x0, x1, y0, y1)]

    def readData(self, name, cutout=True):
        """
        Reads FITS file, mostly from KECK instruments.
        If there are multiple HDUs, the image is assembled according to 
        the kewyrods DETSEC and DATASEC.
        Otherwise hdus[0].data is returned.
        
        If cutout is TRUE, then only the none-zero portion is returned.
        """

        def fix(x0, x1, lim):
            if x1 < x0:
                if x0 >= lim:
                    dx = x0 - x1
                    x0 = lim
                    x1 = x0 - dx
                if x1 <= 1:
                    x0 = x0 - x1
                    x1 = 0
                    return x0, None, -1
                return x0 - 1, x1 - 2, -1
            return x0 - 1, x1, 1

        def match(s0, s1, d0, d1):
            ds = s1 - s0
            dd = d1 - d0
            if (ds > 0) == (dd > 0):
                return d0, d0 + ds
            else:
                return d0, d0 - ds

        def check(x):
            return 0 if x is None else x

        def get_binning(b):
            if b is None:
                return 1, 1
            return [int(x) for x in b.split(",")]

        def apply_binning(xy, binx, biny):
            x0, x1, y0, y1 = xy
            return x0 // binx, x1 // binx, y0 // biny, y1 // biny

        def readMosaic():
            det_size = None
            imgBuf = None
            minx, miny, maxx, maxy = 1e9, 1e9, -1, -1
            binx, biny = 1, 1
            height, width = 0, 0
            for hdu in hdus:

                binning = hdu.header.get("BINNING")
                if binning is not None:
                    binx, biny = get_binning(binning)

                if imgBuf is None:
                    ds = hdu.header.get("DETSIZE")
                    if ds is None:
                        continue
                    det_size = self._split_size(ds)
                    x0, x1, y0, y1 = apply_binning(det_size, binx, biny)
                    imgBuf = np.zeros((y1 - y0 + 1, x1 - x0 + 1))
                    height, width = imgBuf.shape

                if hdu.data is None:
                    continue
                detsec = hdu.header.get("DETSEC")
                datasec = hdu.header.get("DATASEC")
                if detsec is None or datasec is None:
                    continue
                dst_x0, dst_x1, dst_y0, dst_y1 = self._split_size(detsec)
                src_x0, src_x1, src_y0, src_y1 = self._split_size(datasec)

                dst_x0, dst_x1 = match(src_x0, src_x1, dst_x0 // binx, dst_x1 // binx)
                dst_y0, dst_y1 = match(src_y0, src_y1, dst_y0 // biny, dst_y1 // biny)
                hduh, hduw = hdu.data.shape

                dstx_0, dstx_1, dstx_inc = fix(dst_x0, dst_x1, width)
                dsty_0, dsty_1, dsty_inc = fix(dst_y0, dst_y1, height)

                srcx_0, srcx_1, srcx_inc = fix(src_x0, src_x1, width)
                srcy_0, srcy_1, srcy_inc = fix(src_y0, src_y1, height)

                dstx_1p = check(dstx_1)
                dsty_1p = check(dsty_1)

                dstx_0p = (dstx_0 + 1) if dstx_0 > dstx_1p else dstx_0

                minx = int(np.min((dstx_0p, dstx_1p, minx)))
                miny = int(np.min((dsty_0, dsty_1p, miny)))
                maxx = int(np.max((dstx_0p, dstx_1p, maxx)))
                maxy = int(np.max((dsty_0, dsty_1p, maxy)))

                dstx_slice = slice(dstx_0, dstx_1, dstx_inc)
                dsty_slice = slice(dsty_0, dsty_1, dsty_inc)
                srcx_slice = slice(srcx_0, srcx_1, srcx_inc)
                srcy_slice = slice(srcy_0, srcy_1, srcy_inc)

                imgBuf[dsty_slice, dstx_slice] = hdu.data[srcy_slice, srcx_slice]
            if cutout:
                imgBuf = imgBuf[miny:maxy, minx:maxx]
            return imgBuf

        def readSingle():
            data = hdus[0].data
            return data

        with open_nowarning(name) as hdus:
            if len(hdus) > 1:
                header = hdus[1].header
                if header.get("DETSIZE") is not None:
                    return readMosaic()
            return readSingle()
