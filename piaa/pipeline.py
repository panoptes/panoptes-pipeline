import numpy as np
import numpy.ma as ma

from astropy.stats import SigmaClip
from astropy import units as u

from photutils import Background2D, MedianBackground, BkgZoomInterpolator, make_source_mask

from pocs.utils import current_time

from piaa.utils.helpers import get_rgb_masks


class BiasSubtract():
    def __init__(self, bias=2048, *args, **kwargs):
        self.bias = bias

    def run(self, data, *args, **kwargs):
        d0 = data - self.bias
        return d0


class DarkCurrentSubtract():
    def __init__(self, sensor_temp, *args, **kwargs):
        self.sensor_temp = (sensor_temp + 273.15) * u.Kelvin

    def run(self, data, *args, **kwargs):
        """ CHANGE ME """
        d0 = data - 10
        return d0


class MaskBadPixels():
    def __init__(self, *args, **kwargs):
        pass

    def run(self, data, max_level=13000, *args, **kwargs):
        """ Currently masks data greater than 13000 """
        return ma.masked_greater_equal(data, max_level, copy=False)


class BackgroundSubtract():
    def __init__(self, *args, **kwargs):
        pass

    def run(self, data, **kwargs):
        return data - self.get_background(data, **kwargs)

    def get_background(self,
                       data,
                       sigma=3,
                       iterations=5,
                       mesh_size=(44, 48),
                       filter_size=(4, 4),
                       dilate_size=12,
                       rgb_masks=None,
                       source_mask=None,
                       estimator=MedianBackground(),
                       interpolator=BkgZoomInterpolator(),
                       verbose=False
                       ):
        """

        Note: mesh_size should be a multiple of them image. Default size
        was based on empirical examination of smooth background. Good
        choices are:
        # Image size: 3476 x 5208
          [
              (316, 434),
              (44, 48),
              (22, 24),
              (11, 12)
          ]

        """
        start_time = current_time()
        rgb_bkgs = dict()
        if verbose:
            print("Getting RGB masks")

        if rgb_masks is None:
            rgb_masks = get_rgb_masks(data)

        if source_mask is None:
            if verbose:
                print("Making source masks")
            source_mask = make_source_mask(
                data,
                snr=3,
                npixels=3,
                sigclip_sigma=sigma,
                sigclip_iters=iterations,
                dilate_size=dilate_size
            )

        if verbose:
            print("\tTime: {:0.2f} seconds\n".format((current_time() - start_time).sec))

        for color, mask in zip(['R', 'G', 'B'], rgb_masks):
            if verbose:
                print("Background channel: ", color)
            bkg_time = current_time()
            sigma_clip = SigmaClip(sigma=sigma, iters=iterations)
            bkg = Background2D(data, mesh_size, filter_size=filter_size,
                               sigma_clip=sigma_clip, bkg_estimator=estimator, mask=np.ma.mask_or(
                                   ~mask, source_mask),
                               interpolator=interpolator, exclude_percentile=80)
            end_time = current_time()
            rgb_bkgs[color] = bkg

            if verbose:
                print("\t{} Background\t Value: {:.02f}\t RMS: {:.02f}".format(
                    color, bkg.background_median, bkg.background_rms_median))
                print("\tTime: {:0.2f} seconds\n".format((end_time - bkg_time).sec))

        if verbose:
            print("Total Time: {:0.2f} seconds\n".format((end_time - start_time).sec))

        r_bkg_data = ma.array(rgb_bkgs['R'].background, mask=~
                              rgb_masks[0]).filled(0).astype(np.int16)
        g_bkg_data = ma.array(rgb_bkgs['G'].background, mask=~
                              rgb_masks[1]).filled(0).astype(np.int16)
        b_bkg_data = ma.array(rgb_bkgs['B'].background, mask=~
                              rgb_masks[2]).filled(0).astype(np.int16)

        return (r_bkg_data + b_bkg_data + g_bkg_data).data


class PanPipeline(object):
    def __init__(self, nodes):
        self.nodes = nodes

    def run(self, data, **kwargs):
        for node in self.nodes:
            data = node.run(data, **kwargs)

        return data
