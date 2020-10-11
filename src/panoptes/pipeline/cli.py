import click

from keckdrpframework.core.framework import Framework
from keckdrpframework.config.framework_config import ConfigClass
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.utils.drpf_logger import getLogger
import subprocess
import time
import argparse
import sys
import traceback
import pkg_resources
import logging.config

from panoptes.pipeline.pipes.main import PanoptesPipeline


@click.command(help='PANOPTES PIPELINE Runner')
@click.option('--config-file', help="Configuration file for pipeline", default='panoptes.cfg')
@click.option('--framework-config-file', help="Configuration file for framework", default='framework.cfg')
@click.option('--framework-logger-file', help="Configuration file for framework logger", default='logger.cfg')
@click.option('--input-file', help='Input image file (full path, list ok)')
@click.option('--fits-files', help="Input FITS files")
@click.option('--image-directory', help="Input image directory containing FITS images.")
@click.option("--ingest_data_only", help="Ingest data and terminate")
@click.option("--wait_for_event", help="Wait for events")
@click.option('--monitor/--no-monitor', default=False, help='Monitor directory for incoming files, default False')
@click.option("--keep-running", default=True, help="Continue processing, wait for ever, default True")
@click.option("--queue_manager_only", help="Starts queue manager only, no processing (useful for RPC)")
def run(config_file=None,
        framework_config_file=None,
        framework_logger_file=None,
        input_file=None,
        fits_files=None,
        image_directory=None,
        queue_manager_only=None,
        ingest_data_only=None,
        wait_for_event=None,
        monitor=False,
        keep_running=True
        ):
    config_namespace = 'panoptes.pipeline.config'

    framework_config_fullpath = pkg_resources.resource_filename(config_namespace, framework_config_file)
    framework_logcfg_fullpath = pkg_resources.resource_filename(config_namespace, framework_logger_file)

    if config_file is None:
        pipeline_config_fullpath = pkg_resources.resource_filename(config_namespace, config_file)
        pipeline_config = ConfigClass(pipeline_config_fullpath, default_section='TEMPLATE')
    else:
        pipeline_config = ConfigClass(config_file, default_section='TEMPLATE')

    # END HANDLING OF CONFIGURATION FILES ##########

    try:
        framework = Framework(PanoptesPipeline, framework_config_fullpath)
        # logging.config.fileConfig(framework_logcfg_fullpath)
        framework.config.instrument = pipeline_config
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)

    # Keep pipeline logger separate from the framework.
    framework.context.pipeline_logger = getLogger(framework_logcfg_fullpath, name="TEMPLATE")
    framework.logger = getLogger(framework_logcfg_fullpath, name="DRPF")

    framework.logger.info(f"PanoptesPipeline initialized")

    try:
        # Ingest image directory, trigger "next_file" on each file.
        if image_directory is not None:
            framework.ingest_data(image_directory, fits_files, monitor)
            framework.start(queue_manager_only, ingest_data_only, wait_for_event, keep_running)
    except KeyboardInterrupt:
        framework.logger.info(f'PanoptesPipeline cancelled. Please wait for shutdown...')
    finally:
        framework.logger.info(f'PanoptesPipeline shutdown')


if __name__ == "__main__":
    run()
