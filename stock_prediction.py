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
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By


# yfinance = YFinance()
# df = yfinance.query(['AAPL'], start_date='3/1/2023', end_date='3/6/2024')

# df['Datetime'] = pd.to_datetime(df['Datetime']).dt.date
class StockPrediction():
    def __init__(self,stock_name):
        self.stock_name=stock_name
        self.stock_data=None
        self.stock_predictions=None
        self.actual_data=None
        self.real_time_data=None
    
    def loading_stock_data(self,start_date,end_date):
        # self.stock_data = yf.download(tickers=self.stock_name, interval='1m', start=start_date,end=end_date)
        # self.stock_data.reset_index(inplace=True)
        # self.stock_data=self.stock_data[["Datetime","Open"]]
        # self.stock_data["Forecasted"]=[0]*len(self.stock_data)
        # self.stock_data["Datetime"]=[datetime.fromisoformat(str(i)).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M') for i in self.stock_data["Datetime"]]
        # self.stock_data.to_csv(f"{self.stock_name}.csv")
        self.stock_data=pd.read_csv("AAPL.csv")
        self.stock_data["Forecasted"]=[0]*len(self.stock_data)
        self.stock_data["p_0.1"]=[0]*len(self.stock_data)
        self.stock_data["p_0.5"]=[0]*len(self.stock_data)
        self.stock_data["p_0.9"]=[0]*len(self.stock_data)


        
        
    def real_time_stock_market_data(self):
        now=datetime.now()
        print("Todays timestamp",now)
        end_time = datetime.combine(now.date(), time(16, 40))
        real_time_stock_data_list=[]
        options={0:"Current Traded Price",1:"Open",2:"Day's Range",3:"52 week range",4:"Volume",5:"Volume",6:"Market Cap",7:"PE Ratio",8:"EPS",9:"1y Target Est"}
        url=f"https://www.marketwatch.com/investing/stock/aapl"
        interval=datetime.now()
        while interval<=end_time:
            interval=datetime.now()
            chromeOptions = webdriver.ChromeOptions()
            chromeOptions.add_argument("window-size=1200x600")
            browser = webdriver.Chrome(options=chromeOptions)
            url=f'https://finance.yahoo.com/quote/AAPL/'
            browser.get(url)
            fin_streamer =browser.find_element("xpath","/html/body/div[1]/main/section/section/section/article/section[1]/div[2]/div[1]/section/div/section/div[1]/fin-streamer[1]")
            span_element = fin_streamer.find_element(By.TAG_NAME, 'span')
            current_price = span_element.get_attribute('innerHTML')
            print("Using device",device)
            data={}
            k=0
            data["Datetime"]=interval.strftime('%Y-%m-%d %H:%M')
            data["Traded Price"]=[current_price]
            
            print("The data",data)
            df=pd.DataFrame(data=data)
            print(df.head())
            df["Traded Price"]=df["Traded Price"].astype(float)
            # df["Close"]=df["Close"].astype(float)
            # df["Volume"]=df["Volume"].apply(lambda x:int(x.replace(",","")))
            print(df.head())
            
            real_time_stock_data_list.append(df)
            self.real_time_data=pd.concat(real_time_stock_data_list)
            # self.real_time_data=self.real_time_data[["Datetime","Open","Close","Volume"]]
            self.real_time_data.to_csv(f"{self.stock_name}_real_time_data.csv",index=False)
            # self.stock_data=pd.concat([self.stock_data,self.real_time_data])
            # self.stock_data.to_csv(f"{self.stock_name}.csv",index=False)
            time_new.sleep(60)
     
    def train_nbeats_model(self):
        print("The device used is",device)
        torch.set_float32_matmul_precision("low")
        vertical=self.stock_data
        LOAD = False        
        EPOCHS = 3
        INLEN = 10
        BLOCKS = 64         
        LWIDTH = 32
        BATCH = 32       
        LEARN = 1e-5        
        VALWAIT = 1         
        N_FC = 1            
        RAND = 42           
        N_SAMPLES = 10 
        N_JOBS = 3          
        QUANTILES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        SPLIT = 0.8 
        qL1, qL2 = 0.01, 0.10        
        qU1, qU2 = 1-qL1, 1-qL2,     
        label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
        label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'
        final_verticals_df=[]
        vertical=vertical.loc[:,vertical.columns.str.contains("Datetime|Open")]
        vertical["Datetime"]=pd.to_datetime(vertical['Datetime'])
        vertical.rename(columns={"Datetime":"time"},inplace=True)
        ts_P = TimeSeries.from_dataframe(vertical,time_col="time",fill_missing_dates=True,freq="1min")
        ts_P=ts_P.pd_dataframe()
        ts_P.to_csv("Values.csv")
        ts_P.fillna(vertical["Open"].values[-1], inplace=True)
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
        last_date=pd.Timestamp(vertical["time"].values[-1])+timedelta(minutes=1)
        future_dates = pd.date_range(start=last_date, periods=50, freq='1min')
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
        data.update({"Datetime":future_dates})
        new_df=pd.DataFrame(data)
        final_verticals_df.append(new_df)
        self.stock_predictions=pd.concat(final_verticals_df)
        self.stock_predictions['Datetime'] = pd.to_datetime(self.stock_predictions['Datetime'])
        columns=[i for i in self.stock_predictions.columns if "mape" in i]
        min1=float("inf")
        column=None
        for i in columns:
            val=self.stock_predictions[i].values[0]
            if val<min1:
                column=i
                min1=val
        column=column.split("_mape")[0]
        # output=output[(output["Datetime"]<pd.to_datetime(f"{start_date} 9:30")) & (output["Datetime"]<pd.to_datetime(f"{start_date} 16:00"))]
        self.stock_predictions.to_csv("predictions.csv")
        if column=="p_0.1":
            output=self.stock_predictions[["Datetime",column,"p_0.5","p_0.9"]]
            output["Open"]=[0]*len(output)
            self.stock_data.drop("p_0.1",axis=1,inplace=True)
            output.rename(columns={column:"Forecasted"},inplace=True)
            self.stock_predictions=pd.concat([self.stock_data,output])
            data={"Datetime":[str(i) for i in self.stock_predictions["Datetime"]],"Open":list(self.stock_predictions["Open"]),"Forecasted":list(self.stock_predictions["Forecasted"]),"p50":list(self.stock_predictions["p_0.5"]),"p90":list(self.stock_predictions["p_0.9"]),"length":len(self.stock_data)}
        elif column=="p_0.5":
            output=self.stock_predictions[["Datetime",column,"p_0.1","p_0.9"]]
            output=output[(output["Datetime"]>=vertical["time"].values[-1])]
            self.stock_data.drop("p_0.5",axis=1,inplace=True)
            output["Open"]=[0]*len(output)
            self.stock_predictions=pd.concat([self.stock_data,output])
            data={"Datetime":[str(i) for i in self.stock_predictions["Datetime"]],"Open":list(self.stock_predictions["Open"]),"Forecasted":list(self.stock_predictions["Forecasted"]),"p10":list(self.stock_predictions["p_0.1"]),"p90":list(self.stock_predictions["p_0.9"]),"length":len(self.stock_data)}
        elif column=="p_0.9":
            output=self.stock_predictions[["Datetime",column,"p_0.5","p_0.1","p_0.9"]]
            output=output[(output["Datetime"]>=vertical["time"].values[-1])]
            self.stock_data.drop("p_0.9",axis=1,inplace=True)
            output["Open"]=[0]*len(output)
            self.stock_predictions=pd.concat([self.stock_data,output])
            data={"Datetime":[str(i) for i in self.stock_predictions["Datetime"]],"Open":list(self.stock_predictions["Open"]),"Forecasted":list(self.stock_predictions["Forecasted"]),"p10":list(self.stock_predictions["p_0.1"]),"p50":list(self.stock_predictions["p_0.5"]),"length":len(self.stock_data)}
        else:
            self.stock_data[column]=[0]*len(self.stock_data)
            output=self.stock_predictions[["Datetime",column,"p_0.1","p_0.5","p_0.9"]]
            output["Open"]=[0]*len(output)
            self.stock_predictions=pd.concat([self.stock_data,output])
            data={"Datetime":[str(i) for i in self.stock_predictions["Datetime"]],"Open":list(self.stock_predictions["Open"]),"Forecasted":list(self.stock_predictions["Forecasted"]),"p10":list(self.stock_predictions["p_0.1"]),"p50":list(self.stock_predictions["p_0.5"]),"p90":list(self.stock_predictions["p_0.9"]),"length":len(self.stock_data)}
        self.stock_predictions.to_csv("predictions.csv")
        
        print(data)
        return data
        
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
# date=datetime.now().date()
# weekday=date.weekday()
# if weekday==0:
    # start_date=date-timedelta(days=5)
    # end_date=start_date+timedelta(days=2)
# elif weekday!=5 and weekday!=6:
# start_date=date-timedelta(days=6)
# end_date=date+timedelta(days=1)
# date_obj = datetime.strptime(end_date, "%Y-%m-%d")
# new_date_obj = date_obj + timedelta(days=2)
# new_date = new_date_obj.strftime("%Y-%m-%d")
# sp=StockPrediction("AAPL")
# sp.loading_stock_data(start_date,end_date)
# sp.real_time_stock_market_data()
# sp.train_nbeats_model()
# # sp.stock_test_plot(end_date,new_date)
# sp.stock_future_plot(end_date,new_date)
