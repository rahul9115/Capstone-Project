import psycopg2
from config import load_config
from datetime import datetime,timedelta
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
            for i in values:
                print("Value",i[2]==self.stock_ticker)
                if i[2]==self.stock_ticker:
                    flag=False
                    break
            if flag==True:
                cursor.execute(query)
                self.conn.commit()
                cursor.close()
                self.conn.close()
            else:
                print("The stock already exists")
        elif table_name=="forecasting_algorithms":
            flag=True
            if len(values)>0:
                for i in values:
                    if i[1]==self.forecasting_algorithm:
                        flag=False
                        break
            print("Here",flag)
            if flag==True:
                cursor.execute(query)
                self.conn.commit()
                cursor.close()
                self.conn.close()
            else:
                print("The algorithm already exists")
        elif table_name=="backtest_simulation":
            flag=True
            print("backtest",values)
            if len(values)>0:
                for i in values:
                    if i[1]==self.training_period:
                        flag=False
                        break
            if flag==True:
                cursor.execute(query)
                self.conn.commit()
                cursor.close()
                self.conn.close()
            else:
                print("The simulation already exists")
        elif table_name=="real_time_forecast_datetime":
            flag=True
            if len(values)>0:
                for i in values:
                    print("yo",str(i[1].strftime('%Y-%m-%d'))==str(self.start_date))
                    if str(i[1].strftime('%Y-%m-%d'))==str(self.start_date):
                        flag=False
                        break
            if flag==True:
                cursor.execute(query)
                self.conn.commit()
                cursor.close()
                self.conn.close()
            else:
                print("The datetime already exists")


           

        # cursor.execute(query)
        # self.conn.commit()
        
    def yfinace_api(self):
        self.stock_data = yf.download(tickers=self.stock_name, interval='1m', start=self.start_date,end=self.end_date, prepost=True)
        self.stock_data.reset_index(inplace=True)
        self.stock_data=self.stock_data[["Datetime","Open"]]
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
        ts_P = TimeSeries.from_dataframe(input_data,time_col="ds",fill_missing_dates=True,freq="1min")
        ts_P=ts_P.pd_dataframe()
        ts_P.fillna(input_data["y"].values[-1], inplace=True)
        forecast_data = nixtla_client.forecast(ts_P, h=100, level=[80, 90])
        print("Forecast data\n",forecast_data)
    
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
            score = mape(ts_test, preds)
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
        future_dates = pd.date_range(start=vertical["time"].values[-1], periods=15, freq='1min')
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
        rtp=pd.read_csv("real_time_predictions.csv")
        rtp=rtp.loc[:,~rtp.columns.str.contains("^Unnamed")]
        real_time_predictions=[rtp]
        time=timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if column=="p_0.1":
            output=self.stock_predictions[["Datetime",column,"p_0.5","p_0.9"]]
            
            output["Open"]=[0]*len(output)
            output.rename(columns={column:"Forecasted"},inplace=True)
            os.makedirs(os.getcwd()+f"\\predictions\\{datetime.now().date()}",exist_ok=True)
            output.to_csv(os.getcwd()+f"\\predictions\\{datetime.now().date()}\\predictions_{time}.csv")
            real_time_predictions.append(output)
            rtp=pd.concat(real_time_predictions)
            rtp.drop_duplicates(inplace=True)
            rtp.to_csv("real_time_predictions.csv",index=False)
            self.stock_predictions=pd.concat([self.stock_data,output])
            
            data={"Datetime":[str(i) for i in self.stock_predictions["Datetime"]],"Open":list(self.stock_predictions["Open"]),"Forecasted":list(self.stock_predictions["Forecasted"]),"p50":list(self.stock_predictions["p_0.5"]),"p90":list(self.stock_predictions["p_0.9"]),"length":len(self.stock_data)}
        elif column=="p_0.5":
            output=self.stock_predictions[["Datetime",column,"p_0.1","p_0.9"]]
            output[["Datetime",column]].to_csv("real_time_predictions.csv")
            output=output[(output["Datetime"]>=vertical["time"].values[-1])]
            output["Open"]=[0]*len(output)
            output.rename(columns={column:"Forecasted"},inplace=True)
            os.makedirs(os.getcwd()+f"\\predictions\\{datetime.now().date()}",exist_ok=True)
            output.to_csv(os.getcwd()+f"\\predictions\\{datetime.now().date()}\\predictions_{time}.csv")
            real_time_predictions.append(output)
            rtp=pd.concat(real_time_predictions)
            # rtp.drop_duplicates(inplace=True)
            rtp.to_csv("real_time_predictions.csv",index=False)
            self.stock_predictions=pd.concat([self.stock_data,output])
            data={"Datetime":[str(i) for i in self.stock_predictions["Datetime"]],"Open":list(self.stock_predictions["Open"]),"Forecasted":list(self.stock_predictions["Forecasted"]),"p10":list(self.stock_predictions["p_0.1"]),"p90":list(self.stock_predictions["p_0.9"]),"length":len(self.stock_data)}
        elif column=="p_0.9":
            output=self.stock_predictions[["Datetime",column,"p_0.5","p_0.1","p_0.9"]]
            output=output[(output["Datetime"]>=vertical["time"].values[-1])]
            output["Open"]=[0]*len(output)
            output.rename(columns={column:"Forecasted"},inplace=True)
            os.makedirs(os.getcwd()+f"\\predictions\\{datetime.now().date()}",exist_ok=True)
            output.to_csv(os.getcwd()+f"\\predictions\\{datetime.now().date()}\\predictions_{time}.csv")
            real_time_predictions.append(output)
            rtp=pd.concat(real_time_predictions)
            rtp.drop_duplicates(inplace=True)
            rtp.to_csv("real_time_predictions.csv",index=False)
            self.stock_predictions=pd.concat([self.stock_data,output])
            data={"Datetime":[str(i) for i in self.stock_predictions["Datetime"]],"Open":list(self.stock_predictions["Open"]),"Forecasted":list(self.stock_predictions["Forecasted"]),"p10":list(self.stock_predictions["p_0.1"]),"p50":list(self.stock_predictions["p_0.5"]),"length":len(self.stock_data)}
        else:
            self.stock_data[column]=[0]*len(self.stock_data)
            output=self.stock_predictions[["Datetime",column,"p_0.1","p_0.5","p_0.9"]]
            output["Open"]=[0]*len(output)
            output.rename(columns={column:"Forecasted"},inplace=True)
            os.makedirs(os.getcwd()+f"\\predictions\\{datetime.now().date()}",exist_ok=True)
            output.to_csv(os.getcwd()+f"\\predictions\\{datetime.now().date()}\\predictions_{time}.csv")
            self.stock_predictions=pd.concat([self.stock_data,output])
            real_time_predictions.append(output)
            rtp=pd.concat(real_time_predictions)
            rtp.drop_duplicates(inplace=True)
            rtp.to_csv("real_time_predictions.csv",index=False)
            data={"Datetime":[str(i) for i in self.stock_predictions["Datetime"]],"Open":list(self.stock_predictions["Open"]),"Forecasted":list(self.stock_predictions["Forecasted"]),"p10":list(self.stock_predictions["p_0.1"]),"p50":list(self.stock_predictions["p_0.5"]),"p90":list(self.stock_predictions["p_0.9"]),"length":len(self.stock_data)}
        self.stock_predictions.to_csv("predictions.csv")
        print(data)
        return data

for i in range(365,-1,-1):
    date=datetime.now().date()
    start_date=date-timedelta(days=i)
    end_date=date+timedelta(days=1)
    training_period="1D"
    bk=Backtesting(None,"Apple Inc","AAPL","NBEATS",start_date,end_date,training_period)
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
    # bk.insert_queries(stock_data_query,"stock_data")
    # bk.insert_queries(forecasting_alg_query,"forecasting_algorithms")
    bk.insert_queries(backtest_simulation_query,"backtest_simulation")
    bk.database_connection()
    bk.insert_queries(real_time_forecast_datetime_query,"real_time_forecast_datetime")
    break
    
