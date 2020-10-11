import logging
import pkg_resources
from configparser import ConfigParser

from keckdrpframework.pipelines.base_pipeline import BasePipeline

from panoptes.utils.logging import logger


class LogInterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[LogInterceptHandler()], level=0)


class PanoptesPipeline(BasePipeline):
    """PANOPTES Image Processing for Exoplanets """

    event_table = {
        "next_file": ("input.read_fits", "file_ready", "file_ready"),
        "file_ready": ("hist_equal2d", "histeq_done", "histeq_done"),
        "histeq_done": ("output.save_image", "output.image.done", "html_list"),
        "html_list": ("output.save_html", None, None),
    }

    def __init__(self, context):
        """
        Constructor
        """
        super().__init__(context)
        self.context = context

        fits2png_config_path = pkg_resources.resource_filename('panoptes.pipeline.config', 'fits2png.cfg')

        fits2png_config = ConfigParser()
        fits2png_config.read(fits2png_config_path)
        self.context.config.fits2png = fits2png_config

        self.cnt = 0
