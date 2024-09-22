import requests
import time as time_new
from datetime import datetime, time

def fetch_data():
    url="http://127.0.0.1:5000/api"
    now=datetime.now()
    end_time = datetime.combine(now.date(), time(16, 40))
    headers = {
    "Authorization": "authorized"
    }
    while True:
        response=requests.get(url,headers=headers)
        time_new.sleep(1200)
fetch_data()