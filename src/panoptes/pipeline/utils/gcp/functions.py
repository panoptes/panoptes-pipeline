from typing import Callable, Any


def cloud_function_entry_point(raw_message: dict,
                               operation: Callable[[str, Any], bool],
                               **kwargs):
    """Generic handler for GCP Cloud Function.

    This method will receive an `objectId` in the `attributes` key of `raw_message`
    which corresponds to the location in the storage bucket. This bucket path
    is called as the first parameter to the `operation` callable, with `attributes`
    also passed by kwargs.

    Args:
        raw_message (dict): The Cloud Functions event payload.
        operation (Callable): The operation to run.
    Returns:
        None; the output is written to Stackdriver Logging
    """
    attributes = raw_message['attributes']
    print(f"Attributes: {attributes!r}")

    bucket_path = attributes['objectId']

    if bucket_path is None:
        raise Exception(f'No file requested')

    output = operation(bucket_path, **attributes, **kwargs)
    if output is None or output is False:
        raise Exception('The process indicated failure but no other information.')

    print(f'Output from processing {bucket_path} {output}')

    return output
