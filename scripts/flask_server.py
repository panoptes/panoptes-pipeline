from flask import Flask, request, render_template, jsonify
import json
import combine_lightcurves


app = Flask(__name__)

@app.route('/', methods=['GET'])
def receive():
	print("GET request received")
	response = {
		'status_code': 200
	}
	pic = "PIC_J0326137+295015"
	combine_lightcurves.write_output("filename.json", "data")
	#exec('combine_lightcurves.py')
	return jsonify(response), 200

if __name__ == '__main__':
	app.run(
		"0.0.0.0",
		8080,
		True
	)
