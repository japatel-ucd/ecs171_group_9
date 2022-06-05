import flask
from flask import request, jsonify
from flask_cors import CORS
import requests
import urllib.request 
from scripts.spectrogram import create_spectrogram
from scripts.call_models import inference

app = flask.Flask(__name__)
CORS(app) # This is required for this backend server running on port 5000 
          # to accept and process requests from another server
          # (in our case, frontend express javascript server running on port 8000)
          # running on a different port. ie., allow cross-origin requests.

# The path where the backend server is listening at

@app.route("/saverecording", methods=["POST"])
def save_recording():
    body = request.json # read the request body
    print(body)

    data = {"success": False}
    try:
        link = ""    
        if "filename" in body: # read user entered data from request body
            link = body["filename"]
            if "unknown" in link:
                data["success"] = False
                data["message"] = "Language not selected"
            else:
                data["success"] = True
                data["src"] = "/spectrograms/" + link[:-4]+".jpg"

                create_spectrogram(link, link)
    except:
        data["success"] = False
        data["message"] = "ERROR"

    return jsonify(data) # return response 

@app.route("/inference", methods=["POST"])
def call_inference():
    body = request.json # read the request body
    print(body)

    data = {"success": False}
    
    link = "" 
  
    if "filename" in body: # read user entered data from request body
        link = body["filename"]
        print(link)
        if "unknown" in link:
            data["success"] = False
            data["message"] = "Language not selected"
        else:
            [g_pred, g_raw, g_res], [d_pred, d_raw, d_res] = inference(link)
            data["success"] = True
            data["google_raw"] = ' {:.4f}'.format(g_raw)
            data["dense_raw"] = ' {:.4f}'.format(d_raw)

            data["google_result"] = g_res
            data["dense_result"] = d_res

            if g_pred:
                data["google_pred"] = " English"
            else:
                data["google_pred"] = " German"
            
            
            if d_pred:
                data["dense_pred"] = " English"
            else:
                data["dense_pred"] = " German"

        #data["google_pred"] = 

    #data["success"] = False
    #data["message"] = "ERROR"

    return jsonify(data) # return response 

# This is the port where this backend server port runs on
if __name__ == '__main__':
    app.run(host= "0.0.0.0", port=5005)