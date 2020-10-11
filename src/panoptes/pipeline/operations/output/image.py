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


class Image(BasePrimitive):
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

        pretty_image_path = self.output_dir + "/" + name.replace(".fits", self.extension)
        image_data = args.image_data
        h, w = image_data.shape
        img1 = np.stack((image_data,) * 3, axis=-1)

        plt.imsave(pretty_image_path, img1, format=self.output_format)

        self.logger.debug(f"Saved {pretty_image_path=}")
        out_args = Arguments(pretty_image_path=pretty_image_path)
        return out_args
