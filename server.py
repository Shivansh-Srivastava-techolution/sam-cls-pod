# Native Library Import
import os
import sys
import traceback
import threading


# Pip or Installed Library Import
from flask import Flask, request
from colorama import Fore, Style


# Custom file Import
from autoai_process import Config
from autoai_process.builtin_func import auto_ai_connect
from autoai_process import train_test_process

print((Fore.RED if Config.ENV != 'prod' else Fore.GREEN) +
      f'Pod Started in {Config.ENV} setting', Style.RESET_ALL)

Config.MODEL_STATUS = 'None'
Config.POD_STATUS = 'Available'

# Creating the app
app = Flask(__name__)


@app.get("/")
def root():
    return {"server running": "true"}


@app.route('/')
def home():
    return "Hello, World! This is the Train-Pod. The pod is active", 200


@app.route('/test', methods=["POST"])
def test_model():
    try:
        print(Config.POD_STATUS, Config.MODEL_STATUS)
        if request.is_json:
            data = request.get_json()
            if Config.POD_STATUS == 'Available':
                Config.MODEL_STATUS = "None"
                p1 = threading.Thread(
                    target=train_test_process.start_test, args=(data,))
                p1.start()
                return "Testing started", 200
            else:
                return "Pod busy", 503
        else:
            return "Request must be JSON", 500
    except Exception:
        print("####################### ERROR #############################")
        print("Error while registering training pod :  ")
        traceback.print_exc(file=sys.stdout)
        print("####################### ERROR #############################")
        return ("Error : %s while queueing model" % 500)



@app.route('/train', methods=["POST"])
def train_model():
    try:
        print(Config.POD_STATUS, Config.MODEL_STATUS)
        if request.is_json:
            data = request.get_json()
            print(data)
            if Config.POD_STATUS == 'Available':
                Config.MODEL_STATUS = "None"
                p1 = threading.Thread(
                    target=train_test_process.start_train, args=(data,))
                p1.start()
                return "Training started", 200
            else:
                return "Pod busy", 503
        else:
            return "Request must be JSON", 500
    except Exception as e:
        print("####################### ERROR #############################")
        print("Error while registering training pod :  ")
        traceback.print_exc(file=sys.stdout)
        print("####################### ERROR #############################")
        return "Error : %s while queueing model" % e, 500


@app.route('/get_update', methods=["GET"])
def get_update():
    try:
        if Config.MODEL_STATUS == 'Training Completed' or Config.MODEL_STATUS == "Testing Completed":
            print(Config.POD_STATUS, Config.MODEL_STATUS)
            Config.POD_STATUS = 'Available'
            Config.MODEL_STATUS = 'None'
            return {'pod_status': Config.POD_STATUS, "model_status": 'Training Completed'}, 200

        else:
            print(Config.POD_STATUS, Config.MODEL_STATUS)
            return {'pod_status': Config.POD_STATUS, 'model_status': Config.MODEL_STATUS}, 200
    except Exception as e:
        print("####################### ERROR #############################")
        print("Error while returning the pod and model status :  ", e)
        print("####################### ERROR #############################")
        return "Error : %s while queueing model" % e, 500


if __name__ == "__main__":
    # Do a Hard Reset
    os.system("curl ifconfig.me")
    os.system("rm -rf sam2_results/*")
    os.system("rm -rf rlef_videos/")
    auto_ai_connect.reset()
    app.run("0.0.0.0", port=8501)
