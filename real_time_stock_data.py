import requests
from datetime import datetime, time
from bs4 import BeautifulSoup
import pandas as pd
now=datetime.now()
end_time = datetime.combine(now.date(), time(00, 45))
import time
real_time_stock_data_list=[]
stock_ticker="AAPL"
options={0:"Previous Close",1:"Open",2:"Day's Range",3:"52 week range",4:"Volume",5:"Avg Volume",6:"Market Cap",7:"PE Ratio",8:"EPS",9:"1y Target Est"}
url=f"https://finance.yahoo.com/quote/{stock_ticker}/?p={stock_ticker}&.tsrc=fin-srch"
interval=datetime.now()
while interval<=end_time:
    interval=datetime.now()
    data={}
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'lxml')
    close_price=soup.find_all('fin-streamer',{'class':"price yf-mgkamr"})
    open_price=soup.find_all('fin-streamer',{'class':"yf-tx3nkj"})
    for price in close_price:
        inner_html = price.decode_contents()
        close_price=inner_html.split("<span>")[1].split("<")[0]
    k=0
    data["Date"]=interval
    data["Close"]=close_price
    for price in open_price:
        inner_html = price.decode_contents()
        data[options.get(k)]=[inner_html.strip()]
        k+=1
    print(data)
    df=pd.DataFrame(data=data)
    print(df.head())
    real_time_stock_data_list.append(df)
    pd.concat(real_time_stock_data_list).to_csv(f"{stock_ticker}_real_time_data.csv")
    time.sleep(60)

    
    
    