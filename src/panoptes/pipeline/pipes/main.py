import pkg_resources
from configparser import ConfigParser

from keckdrpframework.pipelines.base_pipeline import BasePipeline


class PanoptesPipeline(BasePipeline):
    """PANOPTES Image Processing for Exoplanets """

    event_table = {
        "next_file": ("input.fits.ReadFits", "file_ready", "file_ready"),
        "file_ready": ("hist_equal2d", "histeq_done", "histeq_done"),
        "histeq_done": ("output.image.SaveImage", "output.image.done", "html_list"),
        "html_list": ("output.html.SaveHtml", None, None),
    }

    def __init__(self, context):
        """
        Constructor
        """
        super().__init__(context)
        self.context = context

        pipeline_config_path = pkg_resources.resource_filename('panoptes.pipeline.config', 'pipeline.cfg')

        pipeline_config = ConfigParser()
        pipeline_config.read(pipeline_config_path)
        self.context.config.fits2png = pipeline_config

        self.cnt = 0
