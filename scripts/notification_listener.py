from flask import Flask, request, render_template, jsonify
import json
import combine_lightcurves


app = Flask(__name__)

@app.route('/', methods=['POST'])
def receive():
	print("POST request received")
	response = {
		'status_code': 200
	}
	pic = "PIC_J0326137+295015"
	message_str = request.get_data().decode("utf-8")
	message = json.loads(message_str)
	object_name = message['object_name']
	bucket_name = message['bucket_name']
	print(object_name)
	combine_lightcurves.build_mlc(pic)
	#exec('combine_lightcurves.py')
	return jsonify(response), 200

if __name__ == '__main__':
	app.run(
		host="0.0.0.0",
		port=8080,
		debug=True,
		threaded=True
	)
