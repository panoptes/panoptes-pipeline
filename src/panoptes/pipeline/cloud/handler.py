import base64
import sys
from typing import Callable, Any


class Handler(object):
    @staticmethod
    def cloud_function_entry_point(raw_message: dict, context: Any,
                                   operation: Callable[[str, Any], bool]):
        """Background Cloud Function to be triggered by Cloud Storage.

        This will send a pubsub message to a certain topic depending on
        what type of file was uploaded. The services responsible for those
        topics do all the processing.

        Args:
            raw_message (dict): The Cloud Functions event payload.
            context (google.cloud.functions.Context): Metadata of triggering event.
            operation (Callable): The opeartion to run.
        Returns:
            None; the output is written to Stackdriver Logging
        """
        try:
            message = base64.b64decode(raw_message['data']).decode('utf-8')
            attributes = raw_message['attributes']
            print(f"Attributes: {attributes!r}")

            bucket_path = attributes['objectId']

            if bucket_path is None:
                raise Exception(f'No file requested')

            success = operation(bucket_path, **attributes)
            if success is False:
                raise Exception('The process indicated failure but no other information.')
        except Exception as e:
            print(f'error: {e}')
        finally:
            # Flush the stdout to avoid log buffering.
            sys.stdout.flush()
