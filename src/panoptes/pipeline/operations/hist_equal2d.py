"""
Example of a primitive, a subclass of BasePrimitive.

This class applies the histogram equalization to an image.
Created on Jul 9, 2019
        
@author: skwok
"""

import math
import numpy as np

from keckdrpframework.models.arguments import Arguments
from keckdrpframework.primitives.base_primitive import BasePrimitive


class HistEqual2d(BasePrimitive):
    """
    Histogram equalization.
    Example of a primitive.    
    The configuration for this primitive is in the auxiliary configuration file fits2png.cfg.
    """

    def __init__(self, action, context):
        """
        Initializes the superclass and retrieves the configuration parameters or sets defaults.
        """
        BasePrimitive.__init__(self, action, context)
        cfg = self.config.fits2png
        self.n_hist = eval(cfg.get("DEFAULT", "hist_equal_length", fallback="256 * 256"))
        cut_width = cfg.getint("DEFAULT", "hist_equal_cut_width", fallback=3)
        self.cut_low = cfg.getfloat("DEFAULT", "hist_equal_cut_low", fallback=cut_width)
        self.cut_high = cfg.getfloat("DEFAULT", "hist_equal_cut_high", fallback=cut_width)
        self.t_factor = cfg.getfloat("DEFAULT", "hist_equal_t_factor", fallback=5)

    def _remap(self, arr, from_lo, from_hi, to_lo, to_hi):
        if from_hi == from_lo:
            a = np.empty_like(arr)
            a.fill(np.int_(to_lo))
            return a
        else:
            m = (to_hi - to_lo) / (from_hi - from_lo)
            b = -m * from_lo + to_lo
            return np.clip(np.int_(arr * m + b), to_lo, to_hi)

    def _centroid(self, data):
        """
        One step 1D centroiding algo.
        Returns centroid position and standard deviation
        """
        l = len(data)
        ixs = np.arange(l)
        ixs2 = ixs * ixs
        sumarr = np.sum(data)
        if sumarr == 0:
            return l / 2, l
        cen = np.dot(data, ixs) / sumarr
        var = np.dot(data, ixs2) / sumarr - cen * cen
        return cen, math.sqrt(max(0, var))

    def _centroidLoop(self, arr, tol=1, nloop=10, sFactor=1):
        lastCen, cstd = self._centroid(arr)
        cen = lastCen
        alen = arr.shape[0]
        x0, x1 = 0, alen
        for i in range(nloop):
            width = int(cstd * 5 * sFactor)
            width = max(10, width)
            half = width // 2
            x0 = int(cen - half)
            x0 = max(0, min(alen - width, x0))
            x1 = x0 + width
            x1 = max(0, min(alen, x1))
            cen, cstd = self._centroid(arr[x0:x1])
            cen += x0
            # print(i, cen, cstd, x0, x1)
            diff = abs(cen - lastCen)
            if diff < tol:
                return cen, cstd
            lastCen = cen
        return cen, cstd

    def _applyAHEqHelper(self, data, leng, from_lo, from_hi, to_lo, to_hi, n_hist, thold):
        """
        Adaptive histogram equalization
        """
        data1 = self._remap(data, from_lo, from_hi, to_lo, to_hi)
        histg, edges = np.histogram(data1, bins=n_hist, density=False)
        histg[0] = 0
        histg[-1] = 0
        sumb4 = np.sum(histg)
        histg = np.clip(histg, 0, thold)
        hsum = np.cumsum(histg)
        # ramp = np.linspace(0, (sumb4 - hsum[-1]), num=n_hist, endpoint=False)
        # hsum += ramp
        hsum = self._remap(hsum, hsum[0], hsum[-1], 0, 255)
        return hsum[np.int_(data1)]

    def _applyAHEC(self, img):
        n_hist = self.n_hist
        flatData = img.flatten()
        leng = len(flatData)
        histg, edges = np.histogram(flatData, bins=n_hist, density=False)
        histg[0] = 0
        histg[-1] = 0
        cen, cstd = self._centroid(histg)
        lo_idx = int(max(0, cen - self.cut_low * cstd))
        hi_idx = int(min(cen + self.cut_high * cstd, n_hist - 1))

        from_lo = edges[lo_idx]
        from_hi = edges[hi_idx]
        self.cen = cen
        self.stdev = cstd

        sum1 = np.sum(histg[lo_idx:hi_idx])
        thold = sum1 / (hi_idx - lo_idx) * self.t_factor
        thold = max(leng / n_hist, thold)
        self.logger.debug(f"min thold={leng/n_hist:.2f}, sum1={sum1:.1f}, diff={(hi_idx-lo_idx):.0f}")

        self.logger.debug(f"Hist eq. lo={from_lo:.1f}, hi={from_hi:.1f}, cen={cen:.0f}, std={cstd:.1f}, thold={thold:.1f}")
        return self._applyAHEqHelper(flatData, leng, from_lo, from_hi, 0, n_hist - 1, n_hist, thold)

    def _applyAHEq(self, img):
        n_hist = self.config.hist_equal_length
        flatData = img.flatten()
        leng = len(flatData)
        histg, edges = np.histogram(flatData, bins=n_hist, density=False)
        from_lo = edges[0]
        from_hi = edges[-1]
        thold = leng / n_hist
        return self._applyAHEqHelper(flatData, leng, from_lo, from_hi, 0, n_hist - 1, n_hist, thold)

    def _perform(self):
        args = self.action.args
        img = args.img
        h, w = img.shape

        out_args = Arguments()
        out_args.name = args.name
        out_args.img = self._applyAHEC(img).reshape((h, w)).astype(dtype="uint8")
        return out_args
