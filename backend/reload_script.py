import requests
import time as time_new
from datetime import datetime, time, timedelta
import pandas as pd

def fetch_data():
    url="http://127.0.0.1:5000/api"
    now=datetime.now()-timedelta(days=2)
    print(now)
    end_time = datetime.combine(now.date(), time(16, 40))
    headers = {
    "Authorization": "authorized"
    } 
    pd.DataFrame().to_csv("history.csv")
    while True:
        response=requests.get(url,headers=headers)
        time_new.sleep(1200)
fetch_data()