import base64
from typing import Callable, Any

from loguru import logger


def cloud_function_entry_point(raw_message: dict, context: Any,
                               operation: Callable[[str, Any], bool]):
    """Generic handler for GCP Cloud Function.

    This method

    Args:
        raw_message (dict): The Cloud Functions event payload.
        context (google.cloud.functions.Context): Metadata of triggering event.
        operation (Callable): The operation to run.
    Returns:
        None; the output is written to Stackdriver Logging
    """
    try:
        message = base64.b64decode(raw_message['data']).decode('utf-8')
        attributes = raw_message['attributes']
        logger.info(f"Attributes: {attributes!r}")

        bucket_path = attributes['objectId']

        if bucket_path is None:
            raise Exception(f'No file requested')

        success = operation(bucket_path, **attributes)
        if success is False:
            raise Exception('The process indicated failure but no other information.')
        logger.success(f'Processed {bucket_path}')
    except Exception as e:
        logger.error(f'error: {e}')
