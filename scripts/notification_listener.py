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
	message_str = request.get_data()
	message = json.loads(message_str)
	object_name = message['object_name']
	bucket_name = message['bucket_name']
	#combine_lightcurves.write_output("filename.json", "data")
	#exec('combine_lightcurves.py')
	return jsonify(response), 200

if __name__ == '__main__':
	app.run(
		"0.0.0.0",
		8080,
		True
	)
