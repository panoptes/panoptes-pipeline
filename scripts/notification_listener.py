from flask import Flask, request, render_template, jsonify, g
import json
#from scripts.combine_lightcurves import LightCurveCombiner
from scripts import combine_lightcurves
from pocs.utils.google.storage import PanStorage
from oauth2client.client import GoogleCredentials
from httplib2 import Http
#import google.appengine.runtime.DeadlineExceededError as DeadlineExceededError
import time
import subprocess


# Run on GCE instance (image-analysis) using screen, then: python3 notification_listener.py
app = Flask(__name__)


@app.route('/', methods=['POST'])
def receive():
    """Receive a notification via the App Engine proxy when a new light curve is uploaded."""
    g.start = time.time()
    print("POST request received.")

    message_str = request.get_data().decode("utf-8")
    message = json.loads(message_str)
    object_name = message['object_name']
    print("Object exists notification received: {}.".format(object_name))

    pic = None
    for dirname in object_name.split('/'):
        if dirname.startswith("PIC"):
            pic = dirname
    if pic is None:
        response = "Error: PIC not detected from {}.".format(object_name)

    subprocess.Popen(['python3', 'combine_lightcurves.py', pic])
    
    # else:
    #     try:
    #         subprocess.Popen(['python3', 'combine_lightcurves.py', pic])
    #         response = "Master light curve for {} successfully updated.".format(pic)
    #     except Exception as err:
    #         print("Exception occurred while attempting to update light curves: {}".format(err))
    #         response = "Failed to update master light curve for {}.".format(pic)
    #
    # print(response)
    return jsonify(response="Received POST request"), 200


# @app.teardown_request
# def teardown_request(exception=None):
#     diff = time.time() - g.start
#     print("Time to return: {}".format(diff))
#     message_str = request.get_data().decode("utf-8")
#     pic = None
#     message = json.loads(message_str)
#     object_name = message['object_name']
#     print("Object exists notification received: {}.".format(object_name))
#     for dirname in object_name.split('/'):
#         if dirname.startswith("PIC"):
#             pic = dirname
#     if pic is None:
#         response = "Error: PIC not detected from {}.".format(object_name)
#     else:
#         pan_storage = PanStorage(bucket_name='panoptes-simulated-data')
#         combiner = LightCurveCombiner(storage=pan_storage)
#         try:
#             combiner.run(pic)
#             response = "Master light curve for {} successfully updated.".format(pic)
#         except Exception as err:
#             print("Exception occurred while attempting to update light curves: {}".format(err))
#             response = "Failed to update master light curve for {}.".format(pic)
#     print(response)


if __name__ == '__main__':
    print('Listening for notification from App Engine...')
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True,
        threaded=True
    )
