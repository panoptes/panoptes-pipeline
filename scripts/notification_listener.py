from flask import Flask, request, render_template, jsonify
import json
import combine_lightcurves


# Run on GCE instance (image-analysis) using screen, then: python3 notification_listener.py
app = Flask(__name__)

@app.route('/', methods=['POST'])
def receive():
    """Receive a notification via the App Engine proxy when a new light curve is uploaded."""
    print("POST request received")
    pic = None
    message_str = request.get_data().decode("utf-8")
    message = json.loads(message_str)
    object_name = message['object_name']
    print("Object exists notification received: {}.".format(object_name))
    
    for dirname in object_name.split('/'):
        if dirname.startswith("PIC"):
            pic = dirname
    if pic is None:
        print("Error: PIC not detected from {}.".format(object_name))
    else:
        combine_lightcurves.build_mlc(pic)
        print("Master light curve for {} successfully updated.".format(pic))
    return jsonify(object_name=message['object_name'], bucket_name=message['bucket_name']), 200


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True,
        threaded=True
    )
