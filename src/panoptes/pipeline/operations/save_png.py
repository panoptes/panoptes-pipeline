"""
Example of a primitive.
This class saves an image as PNG or JPG, as specified in the configuration.

Created on Jul 8, 2019
                
@author: skwok
"""

import numpy as np
import os
import os.path

from keckdrpframework.models.arguments import Arguments
from keckdrpframework.primitives.base_primitive import BasePrimitive
import matplotlib.pyplot as plt


class SavePng(BasePrimitive):
    def __init__(self, action, context):
        """
        Initializes super class.
        Gets paramenters from configuration.

        """
        BasePrimitive.__init__(self, action, context)

        fcfg = self.config.fits2png
        self.output_dir = self.config.output_directory
        self.extension = fcfg.get("DEFAULT", "output_extension", fallback=".png").strip('"').strip("'")
        self.output_format = fcfg.get("DEFAULT", "output_format", fallback="png").strip('"').strip("'")

    def _perform(self):
        os.makedirs(self.output_dir, exist_ok=True)
        args = self.action.args
        name = os.path.basename(args.name)

        out_name = self.output_dir + "/" + name.replace(".fits", self.extension)
        img = args.img
        h, w = img.shape
        img1 = np.stack((img,) * 3, axis=-1)

        plt.imsave(out_name, img1, format=self.output_format)

        self.logger.debug("Saved {}".format(out_name))
        out_args = Arguments(name=out_name)
        return out_args
