import psycopg2
from config import load_config
from datetime import datetime,timedelta, time
import yfinance as yf
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.metrics import mape, rmse 
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from tspiral.forecasting import ForecastingCascade
from tspiral.model_selection import TemporalSplit
from nixtla import NixtlaClient
import os
import torch 
import optuna
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
load_dotenv()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Backtesting:
    def __init__(self,conn,stock_name,stock_ticker,forecasting_algorithm,start_date,end_date,training_period,news_description=None,news_source=None):
        self.training_period=training_period
        self.start_date=start_date
        self.forecasting_algorithm=forecasting_algorithm
        self.stock_name=stock_name
        self.news_source=news_source
        self.news_description=news_description
        self.conn=conn
        self.end_date=end_date
        self.stock_ticker=stock_ticker
        self.simulation_id=None
        self.date_id=None
        self.alg_id=None
        self.stock_id=None
        self.news_data=None

    def integrate_yfiance_news(self):
        self.stock_data["Datetime"]=pd.to_datetime(self.stock_data["Datetime"])
        merged_data=pd.merge(self.stock_data,self.news_data,on="Datetime",how="left")
        merged_data.reset_index(inplace=True)
        merged_data.to_csv("yfiance_&_news_data.csv")

    def load_news_api(self,start_date,end_date,general=False):
        if general:
            newsapi = NewsApiClient(api_key=os.environ.get("NEWS_API_KEY"))
            all_articles = newsapi.get_everything(q=f'Top USA news',
                                        from_param=str(start_date),
                                        to=str(end_date),
                                        language='en',
                                        sort_by='relevancy',
                                        page=5)
            source_name=[]
            author=[]
            title=[]
            description=[]
            content=[]
            url=[]
            date=[]
            for i in all_articles['articles']:
                source_name.append(i["source"]["name"])
                author.append(i['author'])
                title.append(i["title"])
                content.append(i["content"])
                description.append(i["description"])
                url.append(i["url"])    
                date.append(datetime.strptime(i["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M"))
            self.news_data=pd.DataFrame(data={"Datetime":date,"News Source":source_name,"News author":author,"News Headline":title,"News Description":description,"News URL":url,"News content":content})
            self.news_data.to_csv("news_data.csv",index=False)
            self.news_data.sort_values("Datetime",inplace=True)
            self.news_data['Datetime'] = pd.to_datetime(self.news_data['Datetime'])
            self.news_data.drop_duplicates(subset=["Datetime"],inplace=True,keep="first")
            self.news_data.set_index('Datetime', inplace=True)
            full_datetime_range = pd.date_range(start=self.news_data.index.min(), end=self.news_data.index.max(), freq='min')
            self.news_data = self.news_data.reindex(full_datetime_range, method='ffill')
            self.news_data.reset_index(inplace=True)
            self.news_data.rename(columns={'index': 'Datetime'}, inplace=True)
        else:
            newsapi = NewsApiClient(api_key=os.environ.get("NEWS_API_KEY"))
            all_articles = newsapi.get_everything(q=f'{self.stock_name} news',
                                          from_param=str(start_date),
                                          to=str(end_date),
                                          language='en',
                                          sort_by='relevancy',
                                          page=5)
            source_name=[]
            author=[]
            title=[]
            description=[]
            content=[]
            url=[]
            date=[]
            for i in all_articles['articles']:
                source_name.append(i["source"]["name"])
                author.append(i['author'])
                title.append(i["title"])
                content.append(i["content"])
                description.append(i["description"])
                url.append(i["url"])    
                date.append(datetime.strptime(i["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M"))
            self.news_data=pd.DataFrame(data={"Datetime":date,"News Source":source_name,"News author":author,"News Headline":title,"News Description":description,"News URL":url,"News content":content})
            self.news_data.to_csv("news_data.csv",index=False)
            self.news_data.sort_values("Datetime",inplace=True)
            self.news_data['Datetime'] = pd.to_datetime(self.news_data['Datetime'])
            self.news_data.drop_duplicates(subset=["Datetime"],inplace=True,keep="first")
            self.news_data.set_index('Datetime', inplace=True)
            full_datetime_range = pd.date_range(start=self.news_data.index.min(), end=self.news_data.index.max(), freq='min')
            self.news_data = self.news_data.reindex(full_datetime_range, method='ffill')
            self.news_data.reset_index(inplace=True)
            self.news_data.rename(columns={'index': 'Datetime'}, inplace=True)

    def database_connection(self):
        try:
            params = load_config()
            self.conn = psycopg2.connect(**params)
            print("Connection successful")
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def query_execution(self,query):
        cursor=self.conn.cursor()
        cursor.execute(query)
        self.conn.commit()
        cursor.close()
        self.conn.close()
    
    def insert_queries(self,query,table_name):
        cursor=self.conn.cursor()
        cursor.execute(f"Select * from {table_name};")
        values=cursor.fetchall()
        print("table name",table_name)
        if table_name=="stock_data":
            flag=True
            if len(values)>0:
                for i in values:
                    if i[2]==self.stock_ticker:
                        self.stock_id=i[0]
                        flag=False
                        break
            if flag==True:
                # self.stock_id=int(values[-1][0])+1
                cursor.execute(query)
                self.conn.commit()
                cursor.close()
                self.insert_queries("","stock_data")
                self.conn.close()
            else:
                print("The stock already exists")
        elif table_name=="forecasting_algorithms":
            flag=True
            if len(values)>0:
                for i in values:
                    if i[1]==self.forecasting_algorithm:
                        self.alg_id=i[0]
                        flag=False
                        break
            print("Here",flag)
            if flag==True:
                # self.alg_id=int(values[-1][0])+1
                cursor.execute(query)
                self.conn.commit()
                cursor.close()
                self.insert_queries("","forecasting_algorithms")
                self.conn.close()
            else:
                print("The algorithm already exists")
        elif table_name=="backtest_simulation":
            flag=True
            print("backtest",values)
            if len(values)>0:
                for i in values:
                    if i[1]==self.training_period:
                        self.simulation_id=i[0]
                        flag=False
                        break
            if flag==True:
                # self.simulation_id=int(values[-1][0])+1
                cursor.execute(query)
                self.conn.commit()
                cursor.close()
                self.insert_queries("","backtest_simulation")
                self.conn.close()
            else:
                print("The simulation already exists")
        elif table_name=="real_time_forecast_datetime":
            flag=True
          
            if len(values)>0:
                for i in values:
                    if str(i[1].strftime('%Y-%m-%d'))==str(self.start_date):
                        self.date_id=i[0]
                        print("In here",self.start_date,str(i[1].strftime('%Y-%m-%d')),self.date_id)
                        flag=False
                        break
            if flag==True:
                # self.date_id=int(values[-1][0])+1
                cursor.execute(query)
                self.conn.commit()
                cursor.close()
                self.insert_queries("","real_time_forecast_datetime")
                self.conn.close()
            else:
                print("The datetime already exists")
        elif table_name=="forecasts":
            print("This",self.date_id)
            cursor.execute(query)
            self.conn.commit()
            cursor.close()
            
    def yfinance_api(self,news=False):
        if news:
            self.stock_data = yf.download(tickers=self.stock_ticker, interval='1m', start=self.start_date,end=self.end_date, prepost=True)
            self.stock_data.reset_index(inplace=True)
            self.stock_data=self.stock_data[["Datetime","Open","Close"]]
            self.stock_data["Forecasted"]=[0]*len(self.stock_data)
            self.stock_data["Datetime"]=[datetime.fromisoformat(str(i)).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M') for i in self.stock_data["Datetime"]]
            self.stock_data.to_csv(f"{self.stock_name}.csv")
        else:
            self.stock_data = yf.download(tickers=self.stock_ticker, interval='1m', start=self.start_date,end=self.end_date, prepost=True)
            self.stock_data.reset_index(inplace=True)
            self.stock_data=self.stock_data[["Datetime","Open","Close"]]
            self.stock_data["Forecasted"]=[0]*len(self.stock_data)
            self.stock_data["Datetime"]=[datetime.fromisoformat(str(i)).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M') for i in self.stock_data["Datetime"]]
            self.stock_data.to_csv(f"{self.stock_name}.csv")
            self.stock_data["Forecasted"]=[0]*len(self.stock_data)
            self.stock_data["p_0.1"]=[0]*len(self.stock_data)
            self.stock_data["p_0.5"]=[0]*len(self.stock_data)
            self.stock_data["p_0.9"]=[0]*len(self.stock_data)
    
    def time_gpt_model(self):
        nixtla_client=NixtlaClient(api_key=os.environ.get("NIXTLA_API_KEY"))
        input_data=self.stock_data.rename(columns={"Datetime":'ds',"Open":"y"})
        input_data=input_data[["ds","y"]]
        ts_P = TimeSeries.from_dataframe(input_data,time_col="ds",fill_missing_dates=True,freq="1min")
        ts_P=ts_P.pd_dataframe()
        values=[]
        print("In here")
        for index,value in enumerate(ts_P["y"]):
            if str(value)=="nan":
                values.append(values[index-1])
            else:
                values.append(value)
        ts_P["y"]=values
        forecast_data = nixtla_client.forecast(ts_P, h=100, level=[80, 90])
        print('Check',pd.to_datetime(forecast_data["ds"].values[0]))
        real_time_forecast_datetime_query=f"""
                    insert into real_time_forecast_datetime (datetime) values('{pd.to_datetime(forecast_data["ds"].values[0]).date()}');
                    """
        self.database_connection()
        self.insert_queries(real_time_forecast_datetime_query,"real_time_forecast_datetime")
        self.database_connection()
        accuracy=self.calculate_mape(forecast_data["TimeGPT"].values,ts_P["y"].values)
        for forecast_value,timestamp,timegpt_hi_80,timegpt_hi_90,timegpt_lo_90,timegpt_lo_80,actual in zip(forecast_data["TimeGPT"],forecast_data["ds"],forecast_data["TimeGPT-hi-80"],forecast_data["TimeGPT-hi-90"],forecast_data["TimeGPT-lo-90"],forecast_data["TimeGPT-lo-80"],ts_P["y"]):
                try:
                    forecast_query=f"""
                    insert into forecasts (date_id,simulation_id,timestamp,alg_id,stock_id,forecast_value,accuracy,timegpt_hi_80,timegpt_hi_90,timegpt_lo_80,timegpt_lo_90,actual) values({self.date_id},{self.simulation_id},'{timestamp}',{self.alg_id},{self.stock_id},{forecast_value},{accuracy},{timegpt_hi_80},{timegpt_hi_90},{timegpt_lo_80},{timegpt_lo_90},{actual});
                    """
                    self.insert_queries(forecast_query,"forecasts")
                except Exception as e:
                    print(timestamp)
                    print(e)
                    self.conn.rollback()
                    continue

    def calculate_mape(self,forecasts,actuals):
        sum1=0
        ndz=1e-10
        for i,j in zip(actuals,forecasts):
            print(i,j,np.abs(i-j))
            sum1+=float(np.abs(i-j)/max(i,ndz))
        mape=(sum1*100)/len(actuals)
        return mape
    
    def calculate_mape_time_series(self,actuals,forecasts):
        sum1=0
        forecasts=forecasts.pd_dataframe(forecasts).values
        actuals=actuals.pd_dataframe(actuals).values
        for i,j in zip(actuals[0],forecasts[0]):
            sum1+=float(np.abs(i-j)/i)
        mape=(sum1*100)/len(actuals[0])
        return mape
    
    def train_nbeats_model(self):
        print("The device used is",device)
        from datetime import time
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
       
        vertical_test=vertical[(vertical["Datetime"]>=pd.to_datetime(f"{self.start_date+timedelta(days=1)} 9:45")) & (vertical["Datetime"]<=pd.to_datetime(f"{self.start_date+timedelta(days=1)} 16:00"))]
        vertical_test.to_csv("here.csv")
        vertical=vertical[vertical["Datetime"]<pd.to_datetime(f"{self.start_date+timedelta(days=1)} 9:45")]
        vertical.to_csv("here1.csv")
        vertical.rename(columns={"Datetime":"time"},inplace=True)
        vertical_test.rename(columns={"Datetime":"time"},inplace=True)
        ts_test=TimeSeries.from_dataframe(vertical_test,time_col="time",fill_missing_dates=True,freq="1min")
        ts_test=ts_test.pd_dataframe()
        values=[]
        for index,value in enumerate(ts_test["Open"]):
            if str(value)=="nan":
                values.append(values[index-1])
            else:
                values.append(value)
        ts_test["Open"]=values
        vertical_test=ts_test
        ts_P = TimeSeries.from_dataframe(vertical,time_col="time",fill_missing_dates=True,freq="1min")
        ts_P=ts_P.pd_dataframe()
        ts_P.to_csv("Values.csv")
        values=[]
        for index,value in enumerate(ts_P["Open"]):
            if str(value)=="nan":
                values.append(values[index-1])
            else:
                values.append(value)
        ts_P["Open"]=values
        ts_P = TimeSeries.from_dataframe(ts_P)
        ts_train, ts_test = ts_P.split_after(SPLIT) 
        scalerP = Scaler()
        scalerP.fit_transform(ts_train)
        ts_ttrain = scalerP.transform(ts_train)
        ts_ttest = scalerP.transform(ts_test)    
        ts_t = scalerP.transform(ts_P)
        def objective(trial):
            input_chunk_length = trial.suggest_int('input_chunk_length', 5, 50)  
            output_chunk_length = trial.suggest_int('output_chunk_length', 1, 20)  
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])  
            lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
            model = NBEATSModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                batch_size=batch_size,
                optimizer_kwargs={'lr': lr},
                n_epochs=3,  
                random_state=42
            )
            model.fit(series=ts_ttrain, 
                    val_series=ts_ttest, 
                    verbose=True)
            preds = model.predict(len(ts_ttest))
            score = self.calculate_mape_time_series(ts_test, preds)
            return score
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        best_params = study.best_params
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best MAPE score: {study.best_value}")
        with open("output1.txt","w") as file:
            file.write(str(study.best_params)+" "+str(study.best_value))
        model = NBEATSModel(input_chunk_length=best_params['input_chunk_length'],
                            output_chunk_length=best_params['output_chunk_length'], 
                            num_stacks=BLOCKS,
                            layer_widths=LWIDTH,
                            batch_size=best_params['batch_size'],
                            n_epochs=EPOCHS,
                            nr_epochs_val_period=VALWAIT, 
                            likelihood=QuantileRegression(QUANTILES), 
                            optimizer_kwargs={"lr": best_params['learning_rate']}, 
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
        # last_date=pd.Timestamp(vertical["time"].values[-1])+timedelta(minutes=1)
        print('Check',pd.to_datetime(vertical["time"].values[-1]).date())
        real_time_forecast_datetime_query=f"""
                    insert into real_time_forecast_datetime (datetime) values('{pd.to_datetime(vertical["time"].values[-1]).date()}');
                    """
        self.database_connection()
        self.insert_queries(real_time_forecast_datetime_query,"real_time_forecast_datetime")
        
        future_dates = pd.date_range(start=vertical["time"].values[-1], periods=380, freq='1min')
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
        # rtp=pd.read_csv("real_time_predictions.csv")
        # rtp=rtp.loc[:,~rtp.columns.str.contains("^Unnamed")]
        # real_time_predictions=[rtp]
        # time=timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if column=="p_0.1":
            output=self.stock_predictions[["Datetime",column,"p_0.5","p_0.9"]]
            output["Open"]=[0]*len(output)
            output.rename(columns={column:"Forecasted"},inplace=True)
            output=output[(output["Datetime"]>=pd.to_datetime(f"{self.start_date+timedelta(days=1)} 9:45")) & (output["Datetime"]<=pd.to_datetime(f"{self.start_date+timedelta(days=1)} 16:00"))]
            output.to_csv("predictions.csv")
            print("Here",len(output["Forecasted"].values),len(vertical_test["Open"].values))
            print("Accuracy",self.calculate_mape(list(output["Forecasted"].values),list(vertical_test["Open"].values)))
            accuracy=self.calculate_mape(list(output["Forecasted"].values),list(vertical_test["Open"].values))
            print("Accuracy:",accuracy)
            self.database_connection()
            for forecast_value,timestamp,p_10,p_50,p_90,actual in zip(output["Forecasted"],output["Datetime"],output["Forecasted"],output["p_0.5"],output["p_0.9"],vertical_test["Open"]):
                try:
                    forecast_query=f"""
                    insert into forecasts (date_id,simulation_id,timestamp,alg_id,stock_id,forecast_value,accuracy,p_10,p_50,p_90,actual) values({self.date_id},{self.simulation_id},'{timestamp}',{self.alg_id},{self.stock_id},{forecast_value},{accuracy},{p_10},{p_50},{p_90},{actual});
                    """
                    self.insert_queries(forecast_query,"forecasts")
                except Exception as e:
                    print(timestamp)
                    print(e)
                    self.conn.rollback()
                    continue
            self.conn.close()
            # pd.DataFrame(data={"Actual":vertical_test["Open"].values,"Forecasted":output["Forecasted"].values}).to_csv("mape_calculations.csv")   
        elif column=="p_0.5":
            output=self.stock_predictions[["Datetime",column,"p_0.1","p_0.9"]]
            output[["Datetime",column]].to_csv("real_time_predictions.csv")
            output=output[(output["Datetime"]>=vertical["time"].values[-1])]
            output["Open"]=[0]*len(output)
            output.rename(columns={column:"Forecasted"},inplace=True)
            output.to_csv("predictions.csv")
            output=output[(output["Datetime"]>=pd.to_datetime(f"{self.start_date+timedelta(days=1)} 9:45")) & (output["Datetime"]<=pd.to_datetime(f"{self.start_date+timedelta(days=1)} 16:00"))]
            output.to_csv("predictions.csv")
            print("Here",len(output["Forecasted"].values),len(vertical_test["Open"].values))
            print("Accuracy",self.calculate_mape(list(output["Forecasted"].values),list(vertical_test["Open"].values)))
            accuracy=self.calculate_mape(list(output["Forecasted"].values),list(vertical_test["Open"].values))
            print("Accuracy:",accuracy)
            self.database_connection()
            for forecast_value,timestamp,p_10,p_50,p_90,actual in zip(output["Forecasted"],output["Datetime"],output["p_0.1"],output["Forecasted"],output["p_0.9"],vertical_test["Open"]):
                try:
                    forecast_query=f"""
                    insert into forecasts (date_id,simulation_id,timestamp,alg_id,stock_id,forecast_value,accuracy,p_10,p_50,p_90,actual) values({self.date_id},{self.simulation_id},'{timestamp}',{self.alg_id},{self.stock_id},{forecast_value},{accuracy},{p_10},{p_50},{p_90},{actual});
                    """
                    self.insert_queries(forecast_query,"forecasts")
                except Exception as e:
                    print(timestamp)
                    print(e)
                    self.conn.rollback()
                    continue
            self.conn.close()
            # pd.DataFrame(data={"Actual":vertical_test["Open"].values,"Forecasted":output["Forecasted"].values}).to_csv("mape_calculations.csv")     
        elif column=="p_0.9":
            output=self.stock_predictions[["Datetime",column,"p_0.5","p_0.1"]]
            output=output[(output["Datetime"]>=vertical["time"].values[-1])]
            output["Open"]=[0]*len(output)
            output.rename(columns={column:"Forecasted"},inplace=True)
            output=output[(output["Datetime"]>=pd.to_datetime(f"{self.start_date+timedelta(days=1)} 9:45")) & (output["Datetime"]<=pd.to_datetime(f"{self.start_date+timedelta(days=1)} 16:00"))]
            output.to_csv("predictions.csv")
            print("Here",len(output["Forecasted"].values),len(vertical_test["Open"].values))
            accuracy=self.calculate_mape(list(output["Forecasted"].values),list(vertical_test["Open"].values))
            print("Accuracy:",accuracy)
            self.database_connection()
            for forecast_value,timestamp,p_10,p_50,p_90,actual in zip(output["Forecasted"],output["Datetime"],output["p_0.1"],output["p_0.5"],output["Forecasted"],vertical_test["Open"]):
                try:
                    forecast_query=f"""
                    insert into forecasts (date_id,simulation_id,timestamp,alg_id,stock_id,forecast_value,accuracy,p_10,p_50,p_90,actual) values({self.date_id},{self.simulation_id},'{timestamp}',{self.alg_id},{self.stock_id},{forecast_value},{accuracy},{p_10},{p_50},{p_90},{actual});
                    """
                    self.insert_queries(forecast_query,"forecasts")
                except Exception as e:
                    print(timestamp)
                    print(e)
                    self.conn.rollback()
                    continue
            self.conn.close()
            # pd.DataFrame(data={"Actual":vertical_test["Open"].values,"Forecasted":output["Forecasted"].values}).to_csv("mape_calculations.csv")
        else:
            self.stock_data[column]=[0]*len(self.stock_data)
            output=self.stock_predictions[["Datetime",column,"p_0.1","p_0.5","p_0.9"]]
            output["Open"]=[0]*len(output)
            output.rename(columns={column:"Forecasted"},inplace=True)
            output=output[(output["Datetime"]>=pd.to_datetime(f"{self.start_date+timedelta(days=1)} 9:45")) & (output["Datetime"]<=pd.to_datetime(f"{self.start_date+timedelta(days=1)} 16:00"))]
            output.to_csv("predictions.csv")
            accuracy=self.calculate_mape(list(output["Forecasted"].values),list(vertical_test["Open"].values))
            print("Accuracy:",accuracy)
            self.database_connection()
            for forecast_value,timestamp,p_10,p_50,p_90,actual in zip(output["Forecasted"],output["Datetime"],output["p_0.1"],output["p_0.5"],output["p_0.9"],vertical_test["Open"]):
                try:
                    forecast_query=f"""
                    insert into forecasts (date_id,simulation_id,timestamp,alg_id,stock_id,forecast_value,accuracy,p_10,p_50,p_90,actual) values({self.date_id},{self.simulation_id},'{timestamp}',{self.alg_id},{self.stock_id},{forecast_value},{accuracy},{p_10},{p_50},{p_90},{actual});
                    """
                    self.insert_queries(forecast_query,"forecasts")
                except Exception as e:
                    print(timestamp)
                    print(e)
                    self.conn.rollback()
                    continue
            self.conn.close()
            # pd.DataFrame(data={"Actual":vertical_test["Open"].values,"Forecasted":output["Forecasted"].values}).to_csv("mape_calculations.csv")

date=datetime.now().date()
start_date=date-timedelta(days=7)
end_date = start_date+timedelta(days=1)
start_weekday=start_date.weekday()
end_weekday=end_date.weekday() 
training_period="1D"       
bk=Backtesting(None,"Apple Inc","AAPL","Time GPT",start_date,date,training_period)
bk.load_news_api(start_date-timedelta(days=23),date,False)
bk.yfinance_api(True)
bk.integrate_yfiance_news()
exit()
# exit()
# training_period="1D"
# bk=Backtesting(None,"Apple Inc","AAPL","NBEATS",start_date,end_date,training_period)
# stock_data_query=f"""
# insert into stock_data (stock_name,stock_ticker) values('{bk.stock_name}','{bk.stock_ticker}');
# """
# forecasting_alg_query=f"""
# insert into forecasting_algorithms (name) values('{bk.forecasting_algorithm}');
# """
# backtest_simulation_query=f"""
# insert into backtest_simulation (training_period) values('{bk.training_period}');
# """
# real_time_forecast_datetime_query=f"""
# insert into real_time_forecast_datetime (datetime) values('{bk.start_date}');
# """

# bk.database_connection()
# bk.insert_queries(stock_data_query,"stock_data")
# bk.database_connection()
# bk.insert_queries(forecasting_alg_query,"forecasting_algorithms")
# bk.database_connection()
# bk.insert_queries(backtest_simulation_query,"backtest_simulation")
# bk.yfinance_api()
# bk.time_gpt_model()
# bk.database_connection()
# bk.insert_queries(real_time_forecast_datetime_query,"real_time_forecast_datetime")
# exit()
for stock_name,stock_ticker in zip(["Apple Inc","Amazon","Alphabet Inc.","Tesla Inc"],["AAPL","AMZN","GOOG","TSLA"]):
    for i in range(26,-1,-1):
        try:
            date=datetime.now().date()
            start_date=date-timedelta(days=i)
            end_date = start_date+timedelta(days=2)
            start_weekday=start_date.weekday()
            end_weekday=end_date.weekday()
            print("Start Date",start_date)
            
            if start_weekday==0 or start_weekday==6:
                continue
            training_period="1D"
            bk=Backtesting(None,stock_name,stock_ticker,"Nbeats",start_date,end_date,training_period)
            stock_data_query=f"""
            insert into stock_data (stock_name,stock_ticker) values('{bk.stock_name}','{bk.stock_ticker}');
            """
            forecasting_alg_query=f"""
            insert into forecasting_algorithms (name) values('{bk.forecasting_algorithm}');
            """
            backtest_simulation_query=f"""
            insert into backtest_simulation (training_period) values('{bk.training_period}');
            """
            real_time_forecast_datetime_query=f"""
            insert into real_time_forecast_datetime (datetime) values('{bk.start_date}');
            """
            bk.database_connection()
            bk.insert_queries(stock_data_query,"stock_data")
            bk.database_connection()
            bk.insert_queries(forecasting_alg_query,"forecasting_algorithms")
            bk.database_connection()
            bk.insert_queries(backtest_simulation_query,"backtest_simulation")
            bk.database_connection()
            bk.insert_queries(real_time_forecast_datetime_query,"real_time_forecast_datetime")
            # exit()
            try:
                bk.yfinance_api()
            except Exception as e:
                continue
            bk.train_nbeats_model()

            training_period="1D"
            bk=Backtesting(None,stock_name,stock_ticker,"Time GPT",start_date,end_date,training_period)
            stock_data_query=f"""
            insert into stock_data (stock_name,stock_ticker) values('{bk.stock_name}','{bk.stock_ticker}');
            """
            forecasting_alg_query=f"""
            insert into forecasting_algorithms (name) values('{bk.forecasting_algorithm}');
            """
            backtest_simulation_query=f"""
            insert into backtest_simulation (training_period) values('{bk.training_period}');
            """
            real_time_forecast_datetime_query=f"""
            insert into real_time_forecast_datetime (datetime) values('{bk.start_date}');
            """
            bk.database_connection()
            bk.insert_queries(stock_data_query,"stock_data")
            bk.database_connection()
            bk.insert_queries(forecasting_alg_query,"forecasting_algorithms")
            bk.database_connection()
            bk.insert_queries(backtest_simulation_query,"backtest_simulation")
            bk.database_connection()
            bk.insert_queries(real_time_forecast_datetime_query,"real_time_forecast_datetime")
            try:
                bk.yfinance_api()
            except Exception as e:
                continue
            bk.time_gpt_model()
        except:
            continue
       
   
        
    
