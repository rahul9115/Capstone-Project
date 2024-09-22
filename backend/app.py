from flask import Flask,Response, jsonify, redirect, url_for, request
from flask_cors import CORS
import pandas as pd
from stock_prediction import *
import time as time_new
import json
import random
app = Flask(__name__)
CORS(app)
executed=False
# @app.route("/")
# def index():
    
#     interval = datetime.now()
#     end_time = interval + timedelta(hours=1)  # Assuming end_time is 1 hour later
#     while interval <= end_time:
#         interval = datetime.now()
#         return redirect(url_for("stream_stock_data"))
        
@app.route('/api', methods=['GET'])
def stream_stock_data():
    print("here",request.headers.get("Authorization"))
    if request.headers.get("Authorization")=="authorized":
        start_date = datetime.now() - timedelta(days=5)
        end_date = datetime.now() + timedelta(days=1)
        sp = StockPrediction("AAPL")
        sp.loading_stock_data(start_date, end_date)
        data = sp.train_nbeats_model()
        with open("executed.txt","w") as file:
            file.write("True")
        with open("data.json","w") as file:
            json.dump(data,file)
    
        # First render (immediate response)
        # print(jsonify())
        print(dict(data).keys())
        
        # data = sp.train_nbeats_model()
        print("The",data)
        
        return jsonify(data)
    else:
        with open("data.json","r") as file:
            data=json.load(file)
        
        return jsonify(data)
    
    # time_new.sleep(120)
# After the first render, wait for 15 minutes for the next update
        # Sleep for 15 minutes (900 seconds)
    
                # stream_stock_data()
    

# def generate_stock_data():
#     while True:
#         data = {
#             "timestamp":time_new.time(),
#             "value":random.random()
#         }
#         yield f"{json.dumps(data)}"
#         time_new.sleep(10)
    

# @app.route('/api/data', methods=['GET'])
# def stream_stock_data():
#     return Response(generate_stock_data(),content_type="text/event-stream")

if __name__ == '__main__':
    app.run(debug=True, threaded=True)