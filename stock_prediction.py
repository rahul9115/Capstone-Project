import pandas as pd
import numpy as np
from pybroker import YFinance
import pybroker
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.metrics import mape, rmse 
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from tspiral.forecasting import ForecastingCascade
from tspiral.model_selection import TemporalSplit
import matplotlib.pyplot  as plt
import pandas as pd
from pybroker import YFinance
yfinance = YFinance()
import matplotlib.dates as mdates
import yfinance as yf
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
pybroker.enable_data_source_cache('yfinance')
import pandas as pd
import yfinance as yf
import torch
from datetime import datetime, timedelta
import requests
from datetime import datetime, time
from bs4 import BeautifulSoup
import pandas as pd
import time as time_new
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device",device)
# yfinance = YFinance()
# df = yfinance.query(['AAPL'], start_date='3/1/2023', end_date='3/6/2024')

# df['Datetime'] = pd.to_datetime(df['Datetime']).dt.date
class StockPrediction():
    def __init__(self,stock_name):
        self.stock_name=stock_name
        self.stock_data=None
        self.stock_predictions=None
        self.actual_data=None
    
    def loading_stock_data(self,start_date,end_date):
        self.stock_data = yf.download(tickers=self.stock_name, interval='1m', start=start_date,end=end_date)
        self.stock_data.reset_index(inplace=True)
        self.stock_data.to_csv(f"{self.stock_name}.csv")
        
    def real_time_stock_market_data(self):
        now=datetime.now()
        end_time = datetime.combine(now.date(), time(00, 45))
        real_time_stock_data_list=[]
        options={0:"Previous Close",1:"Open",2:"Day's Range",3:"52 week range",4:"Volume",5:"Avg Volume",6:"Market Cap",7:"PE Ratio",8:"EPS",9:"1y Target Est"}
        url=f"https://finance.yahoo.com/quote/{self.stock_name}/?p={self.stock_name}&.tsrc=fin-srch"
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
            pd.concat(real_time_stock_data_list).to_csv(f"{self.stock_name}_real_time_data.csv")
            time_new.sleep(60)
        
    def train_nbeats_model(self):
        vertical=self.stock_data
        LOAD = False        
        EPOCHS = 5
        INLEN = 1   
        BLOCKS = 64         
        LWIDTH = 32
        BATCH = 1          
        LEARN = 1e-3        
        VALWAIT = 1         
        N_FC = 1            
        RAND = 42           
        N_SAMPLES = 10 
        N_JOBS = 3          
        QUANTILES = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
        SPLIT = 0.9 
        qL1, qL2 = 0.01, 0.10        
        qU1, qU2 = 1-qL1, 1-qL2,     
        label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
        label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'
        final_verticals_df=[]
        vertical=vertical.loc[:,vertical.columns.str.contains("Datetime|Open")]
        vertical["Datetime"]=pd.to_datetime(vertical['Datetime'])
        vertical.rename(columns={"Datetime":"time"},inplace=True)
        ts_P = TimeSeries.from_dataframe(vertical,time_col="time",fill_missing_dates=True,freq="15min")
        ts_P=ts_P.pd_dataframe()
        ts_P.to_csv("Values.csv")
        ts_P.fillna(0, inplace=True)
        ts_P = TimeSeries.from_dataframe(ts_P)
        ts_train, ts_test = ts_P.split_after(SPLIT) 
        scalerP = Scaler()
        scalerP.fit_transform(ts_train)
        ts_ttrain = scalerP.transform(ts_train)
        ts_ttest = scalerP.transform(ts_test)    
        ts_t = scalerP.transform(ts_P)
        model = NBEATSModel(input_chunk_length=INLEN,
                            output_chunk_length=N_FC, 
                            num_stacks=BLOCKS,
                            layer_widths=LWIDTH,
                            batch_size=BATCH,
                            n_epochs=EPOCHS,
                            nr_epochs_val_period=VALWAIT, 
                            likelihood=QuantileRegression(QUANTILES), 
                            optimizer_kwargs={"lr": LEARN}, 
                            model_name="NBEATS_EnergyES",
                            log_tensorboard=True,
                            generic_architecture=True, 
                            random_state=RAND,
                            force_reset=True,
                            save_checkpoints=True
                        )
        if LOAD:
                pass                           
        else:
            model.fit(series=ts_ttrain, 
                    val_series=ts_ttest, 
                    verbose=True)
        q50_MAPE = np.inf
        ts_q50 = None
        pd.options.display.float_format = '{:,.2f}'.format
        dfY = pd.DataFrame()
        dfY["Actual"] = TimeSeries.pd_series(ts_test)
        metrics={}
        def predQ(ts_t, q):
            ts_tq = ts_t.quantile_timeseries(q)
            ts_q = scalerP.inverse_transform(ts_tq)
            s = TimeSeries.pd_series(ts_q)
            header = "Q" + format(int(q*100), "02d")
            dfY[header] = s
            metrics.update({q:mape(ts_q,ts_test)})
        ts_tpred = model.predict(n=len(ts_ttest),  
                                num_samples=N_SAMPLES,   
                                n_jobs=N_JOBS, 
                                verbose=True)

        _ = [predQ(ts_tpred, q) for q in QUANTILES]
        col = dfY.pop("Q50")
        dfY.insert(1, col.name, col)
        dfY.iloc[np.r_[0:2, -2:0]]
        end = ts_tpred.end_time()
        print("The prediction starting point",vertical["time"].values[-1])
        future_dates = pd.date_range(start=vertical["time"].values[-1], periods=200, freq='15min')
        values1=model.predict(n=len(future_dates),
                        num_samples=N_SAMPLES,  
                        n_jobs=N_JOBS, 
                        verbose=True)
        data={}
        for quantiles in QUANTILES:
            ts_tq = values1.quantile_timeseries(quantiles)
            p10 = scalerP.inverse_transform(ts_tq)
            p10 = p10.values().flatten()
            data.update({f"p_{quantiles}":p10})
            data.update({f"p_{quantiles}_mape":metrics[quantiles]})
        val=list(data.values())[0]
        print(len(val))
        data.update({"date":future_dates})
        new_df=pd.DataFrame(data)
        final_verticals_df.append(new_df)
        self.stock_predictions=pd.concat(final_verticals_df)

        self.stock_predictions.to_csv(f"{self.stock_name}_predictions.csv")
    
    def stock_future_plot(self,start_date,end_date):
        self.stock_predictions=pd.read_csv(f"{self.stock_name}_predictions.csv")
        self.stock_predictions['date'] = pd.to_datetime(self.stock_predictions['date'])
        columns=[i for i in self.stock_predictions.columns if "mape" in i]
        min1=float("inf")
        column=None
        for i in columns:
            val=self.stock_predictions[i].values[0]
            if val<min1:
                column=i
                min1=val
        column=column.split("_mape")[0]
        output=self.stock_predictions
        # output=output[(output["date"]>=pd.to_datetime(start_date)) & (output["date"]<pd.to_datetime(start_date))]
        print(start_date,end_date)
        output=output[(output["date"]>=pd.to_datetime(f"{start_date} 9:00")) & (output["date"]<pd.to_datetime(f"{start_date} 16:00"))]
        plt.plot(output["date"][:-1],output[column][1:],color="b")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.show()
        

    def stock_test_plot(self,start_date,end_date):
        self.actual_data = yf.download([self.stock_name], interval='15m', start=start_date,end=end_date)
        self.actual_data.reset_index(inplace=True)
        self.actual_data['date'] = pd.to_datetime(self.actual_data['Datetime'])
        # self.stock_predictions=pd.read_csv(f"{self.stock_name}_predictions.csv")
        # self.stock_predictions=pd.read_csv(f"output.csv")
        self.stock_predictions['date'] = pd.to_datetime(self.stock_predictions['date'])
        columns=[i for i in self.stock_predictions.columns if "mape" in i]
        min1=float("inf")
        column=None
        
        for i in columns:
            val=self.stock_predictions[i].values[0]
            if val<min1:
                column=i
                min1=val
        # print(self.actual_data)
        # print(self.stock_predictions)
        column=column.split("_mape")[0]
        output=pd.merge(self.actual_data,self.stock_predictions,on="date",how="inner")
        print(output)
        output.reset_index(inplace=True)
        date=output["date"]
        output=output[["Open",column]]
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaled_data = scaler.fit_transform(output)
        output= pd.DataFrame(scaled_data, columns=output.columns)
        output['date']=date
        output=output[(output["date"]>=pd.to_datetime(f"{start_date} 9:00")) & (output["date"]<pd.to_datetime(f"{start_date} 16:00"))]
        plt.plot(output["date"],output["Open"],color="r")
        plt.plot(output["date"][1:],output[column][:-1],color="b")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.show()

start_date="2024-08-13"
end_date="2024-08-15"
date_obj = datetime.strptime(end_date, "%Y-%m-%d")
new_date_obj = date_obj + timedelta(days=2)
new_date = new_date_obj.strftime("%Y-%m-%d")
sp=StockPrediction("AAPL")
sp.loading_stock_data(start_date,end_date)
sp.train_nbeats_model()
# sp.stock_test_plot(end_date,new_date)
sp.stock_future_plot(end_date,new_date)