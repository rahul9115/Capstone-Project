import pandas as pd
import os
import numpy as np
from pybroker import YFinance
import pybroker
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel, NHiTSModel, TiDEModel, TSMixerModel
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
import optuna
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
from nixtla import NixtlaClient
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import httpx
import ssl
import pandas as pd
import csv
import io
import os
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub
import httpx
from langchain_community.utilities import SearchApiAPIWrapper
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
load_dotenv()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
NUM_BLOCKS = 4
LAYER_WIDTHS = 128
POOLING_KERNEL_SIZES = [2, 2, 2]
merged_data_list=[]

# yfinance = YFinance()
# df = yfinance.query(['AAPL'], start_date='3/1/2023', end_date='3/6/2024')

# df['Datetime'] = pd.to_datetime(df['Datetime']).dt.date
class StockPrediction():
    def __init__(self,stock_name,stock_complete_name):
        self.stock_name=stock_name
        self.stock_data=None
        self.stock_predictions=None
        self.actual_data=None
        self.real_time_data=None
        self.stock_complete_name=stock_complete_name
    
    def loading_stock_data(self,start_date,end_date):
        print("Start Date",start_date,"End Date: ",end_date)
        self.stock_data = yf.download(tickers=self.stock_name, interval='1m', start=start_date,end=end_date, prepost=True)
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

    def calculate_mape(self,forecasts,actuals):
            sum1=0
            mape=-1
            forecasts=forecasts.pd_dataframe(forecasts).values
            actuals=actuals.pd_dataframe(actuals).values
            for i,j in zip(actuals[0],forecasts[0]):
                if i==0:
                    continue
                sum1+=float(np.abs(i-j)/i)
            if len(actuals[0])>0:
                mape=(sum1*100)/len(actuals[0])
            return mape
    def select_the_best_model(self,ts_ttrain,ts_ttest,ts_test,scalerP):
        def objective(trial):
            try:
                input_chunk_length = trial.suggest_int('input_chunk_length', 5, 50)
                output_chunk_length = trial.suggest_int('output_chunk_length', 1, 20)
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
                if self.model_name=="NBEATS":
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
                    score = self.calculate_mape(ts_test, scalerP.inverse_transform(preds))
                    return score
                elif self.model_name=="NHiTS":
                    model = NHiTSModel(input_chunk_length=input_chunk_length, 
                                    output_chunk_length=output_chunk_length, 
                                    num_stacks=BLOCKS, 
                                    num_blocks=NUM_BLOCKS,
                                    layer_widths=LAYER_WIDTHS, 
                                    pooling_kernel_sizes=POOLING_KERNEL_SIZES,
                                    batch_size=batch_size, 
                                    n_epochs=3, 
                                    optimizer_kwargs={"lr": lr}, 
                                    random_state=RAND, likelihood=QuantileRegression(QUANTILES)
                                )
                    model.fit(series=ts_ttrain, 
                            val_series=ts_ttest, 
                            verbose=True)
                    preds = model.predict(len(ts_ttest))
                    score = self.calculate_mape(ts_test, scalerP.inverse_transform(preds))
                    return score
                elif self.model_name=="TiDE":
                    model = TiDEModel(input_chunk_length=input_chunk_length, 
                                    output_chunk_length=output_chunk_length, 
                                    hidden_size=LWIDTH,
                                    batch_size=batch_size,
                                    dropout=0.1,
                                    n_epochs=3, 
                                    optimizer_kwargs={"lr": lr}, 
                                    random_state=RAND, 
                                    likelihood=QuantileRegression(QUANTILES)
                                )
                    model.fit(series=ts_ttrain, 
                            val_series=ts_ttest, 
                            verbose=True)
                    preds = model.predict(len(ts_ttest))
                    score = self.calculate_mape(ts_test, scalerP.inverse_transform(preds))
                    return score
                else:
                    model =  TSMixerModel(input_chunk_length=input_chunk_length, 
                                    output_chunk_length=output_chunk_length, 
                                    hidden_size=LWIDTH,
                                    batch_size=batch_size,
                                    dropout=0.1,
                                    n_epochs=3, 
                                    optimizer_kwargs={"lr": lr}, 
                                    random_state=RAND, 
                                    likelihood=QuantileRegression(QUANTILES)
                                )
                    model.fit(series=ts_ttrain, 
                            val_series=ts_ttest, 
                            verbose=True)
                    preds = model.predict(len(ts_ttest))
                    score = self.calculate_mape(ts_test, scalerP.inverse_transform(preds))
                    return score
            except Exception as e:
                print("Its here",e)
        self.models={}
        min1=float("inf")
        for model_name in ["NBEATS","NHiTS","TiDE","TSMixer"]:
            try:
                self.model_name=model_name
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=5)
                best_params = study.best_params
                best_value=study.best_value
                if model_name=="NBEATS":
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
                                    save_checkpoints=True)
                    self.models[model_name]=model
                elif model_name=="NHiTs":
                    model = NHiTSModel(input_chunk_length=best_params['input_chunk_length'], 
                                    output_chunk_length=best_params['output_chunk_length'], 
                                    num_stacks=BLOCKS, 
                                    num_blocks=NUM_BLOCKS,
                                    layer_widths=LAYER_WIDTHS, 
                                    pooling_kernel_sizes=POOLING_KERNEL_SIZES,
                                    batch_size=best_params['batch_size'], 
                                    n_epochs=3, 
                                    optimizer_kwargs={"lr": best_params['learning_rate']}, 
                                    random_state=RAND, 
                                    likelihood=QuantileRegression(QUANTILES)
                                )
                    self.models[model_name]=model
                elif model_name=="TiDE":
                    model = TiDEModel(input_chunk_length=best_params['input_chunk_length'], 
                                    output_chunk_length=best_params['output_chunk_length'], 
                                    num_stacks=BLOCKS, 
                                    num_blocks=NUM_BLOCKS,
                                    layer_widths=LAYER_WIDTHS, 
                                    pooling_kernel_sizes=POOLING_KERNEL_SIZES,
                                    batch_size=best_params['batch_size'], 
                                    n_epochs=3, 
                                    optimizer_kwargs={"lr": best_params['learning_rate']}, 
                                    random_state=RAND, 
                                    likelihood=QuantileRegression(QUANTILES)
                                )
                    self.models[model_name]=model
                else:
                    model = TSMixerModel(input_chunk_length=best_params['input_chunk_length'], 
                                    output_chunk_length=best_params['output_chunk_length'], 
                                    num_blocks=NUM_BLOCKS,
                                    batch_size=best_params['batch_size'], 
                                    n_epochs=3, 
                                    optimizer_kwargs={"lr": best_params['learning_rate']}, 
                                    random_state=RAND, 
                                    likelihood=QuantileRegression(QUANTILES)
                                )
                    self.models[model_name]=model
            except Exception as e:
                print("the model name:",self.model_name,e)
                if model_name=="NBEATS":
                    self.models[model_name]=NBEATSModel(input_chunk_length=INLEN,
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
                elif model_name=="NHiTs":
                    self.models[model_name]=NHiTSModel(input_chunk_length=INLEN, output_chunk_length=N_FC, 
                                                        num_stacks=BLOCKS, num_blocks=NUM_BLOCKS,
                                                        layer_widths=LAYER_WIDTHS, pooling_kernel_sizes=POOLING_KERNEL_SIZES,
                                                        batch_size=BATCH, n_epochs=EPOCHS, optimizer_kwargs={"lr": LEARN}, 
                                                        random_state=RAND, likelihood=QuantileRegression(QUANTILES))

                elif model_name=="TiDE":
                    self.models[model_name]=TiDEModel(input_chunk_length=INLEN, output_chunk_length=N_FC, 
                                                    hidden_size=LWIDTH, dropout=0.1, batch_size=BATCH,
                                                    n_epochs=EPOCHS, optimizer_kwargs={"lr": LEARN}, random_state=RAND,
                                                    likelihood=QuantileRegression(QUANTILES))

                else:
                    self.models[model_name]=TSMixerModel(input_chunk_length=INLEN, output_chunk_length=N_FC,
                                                        hidden_size=LWIDTH, dropout=0.1,
                                                        batch_size=BATCH, n_epochs=EPOCHS, optimizer_kwargs={"lr": LEARN},
                                                        random_state=RAND, likelihood=QuantileRegression(QUANTILES))
                continue
            
        model=self.models["NBEATS"]
        model_final=None
        min1=float("inf")
        for model_name, test_model in self.models.items():
            # try:
            
            test_model.fit(series=ts_ttrain, 
                    val_series=ts_ttest, 
                    verbose=True)
            ts_tpred = test_model.predict(n=len(ts_ttest), num_samples=N_SAMPLES, n_jobs=N_JOBS, verbose=True)
            ts_tpred_rescaled = scalerP.inverse_transform(ts_tpred)
            quantile_mapes = {f"Q{int(q*100)}": self.calculate_mape(ts_test,  scalerP.inverse_transform(ts_tpred_rescaled.quantile_timeseries(q)))
                        for q in QUANTILES}
            avg_mape = np.mean(list(quantile_mapes.values()))
            print("The model_name is",model_name,avg_mape)
            if avg_mape<min1:
                min1=avg_mape
                model_final=model_name
            # except Exception as e:
            #     print("Model selection exception",e)
                # continue
        print("The final is",model_final)
        if model_final:
            model=self.models[model_final]
        else:
            model_final="NBEATS"
            
        return model,model_final

    def ai_agent(self) :
        llm=ChatOpenAI(temperature=0)
        summary_template = """
        Please give me the important current news at this time pertaining to {name_of_stock} company . Don't give me the values
        Get me those as the output.
        """
        search = SearchApiAPIWrapper()
        summary_prompt_template = PromptTemplate(input_variables=["name_of_stock"], template=summary_template)
        open_ai_key = os.environ['OPENAI_API_KEY']
    
        tools_for_agent=[
            Tool(
                name=f"Search for news",
                func=search.run,
                description=f"Useful for getting stock news"
            )
        ]
        react_prompt=hub.pull("hwchase17/react")
        agent=create_react_agent(llm=llm,tools=tools_for_agent,prompt=react_prompt)
        agent_executor=AgentExecutor(agent=agent,tools=tools_for_agent,verbose=True,handle_parsing_errors=True)
        result=agent_executor.invoke(
            input={"input":summary_prompt_template.format_prompt(name_of_stock=self.stock_complete_name)}
        )
        return result

    def sentiment_of_stock_news(self,stock_news):
        print("Stock news",stock_news["output"])
        classifier=pipeline("zero-shot-classification",device=device)
        candidate_labels=["Positive Sentiment","Neutral Sentiment","Negative Sentiment"]
        result=classifier(stock_news["output"],candidate_labels)
        return result["labels"][0]
 
    def train_nbeats_model(self):
        print("The device used is",device)
        torch.set_float32_matmul_precision("low")
        vertical=self.stock_data
        final_verticals_df=[]
        vertical=vertical.loc[:,vertical.columns.str.contains("Datetime|Open")]
        vertical["Datetime"]=pd.to_datetime(vertical['Datetime'])
        vertical.rename(columns={"Datetime":"time"},inplace=True)
        ts_P = TimeSeries.from_dataframe(vertical,time_col="time",fill_missing_dates=True,freq="1min")
        ts_P=ts_P.pd_dataframe()
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
        
        model,model_name=self.select_the_best_model(ts_ttrain,ts_ttest,ts_test,scalerP)
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
        self.stock_predictions.to_csv("predictions.csv")
        real_time_predictions=[]
        time=timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.stock_data["Forecasted"]=[0]*len(self.stock_data)
        output=self.stock_predictions[["Datetime","p_0.1","p_0.5","p_0.9"]]
        output["Open"]=[0]*len(output)
        output["Forecasted"]= self.stock_predictions[column]
        os.makedirs(os.getcwd()+f"\\predictions\\{datetime.now().date()}",exist_ok=True)
        output.to_csv(os.getcwd()+f"\\predictions\\{datetime.now().date()}\\predictions_{time}.csv")
        history=pd.read_csv("history.csv")
        history_data = output.iloc[[1]]
        history=pd.concat([history,history_data])
        history.to_csv("history.csv",index=False)
        self.stock_data.to_csv("stock_data.csv")
        self.stock_predictions=pd.concat([self.stock_data,output])
        real_time_predictions.append(output)
        rtp=pd.concat(real_time_predictions)
        rtp.drop_duplicates(inplace=True)
        rtp.to_csv("real_time_predictions.csv",index=False)
        stock_news=self.ai_agent()
        sentiment=self.sentiment_of_stock_news(stock_news)
        news_df=pd.read_csv("news.csv")
        news_df_merged=pd.concat([news_df,pd.DataFrame(data={"NewsDatetime":[datetime.now().strftime("%Y-%m-%d_%H-%M-%S")],"News":[stock_news["output"]],"Sentiment":[sentiment]})])
        news_df_merged.to_csv("news.csv")
        data={"NewsDatetime":[str(i) for i in news_df_merged["NewsDatetime"]],"News":[str(i) for i in news_df_merged["News"]],"Sentiment":[str(i) for i in news_df_merged["Sentiment"]],"HDatetime":[str(i) for i in history["Datetime"]],"History":list(history["Forecasted"]),"Datetime":[str(i) for i in self.stock_predictions["Datetime"]],"Open":list(self.stock_predictions["Open"]),"Forecasted":list(self.stock_predictions["Forecasted"]),"p10":list(self.stock_predictions["p_0.1"]),"p50":list(self.stock_predictions["p_0.5"]),"p90":list(self.stock_predictions["p_0.9"]),"length":len(self.stock_data),"Best Model":[model_name for _ in range(len((self.stock_predictions)))]}
        self.stock_predictions.to_csv("predictions.csv")
    
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
        output=output[(output["date"]>=pd.to_datetime(f"{datetime.now().date()} 9:00")) & (output["date"]<pd.to_datetime(f"{datetime.now().date()} 16:00"))]
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
# start_date=date-timedelta(days=2)
# end_date=start_date+timedelta(days=1)


# date_obj = datetime.strptime(end_date, "%Y-%m-%d")
# new_date_obj = date_obj + timedelta(days=2)
# new_date = new_date_obj.strftime("%Y-%m-%d")
# sp=StockPrediction("AAPL")
# sp.loading_stock_data(start_date,end_date)
# sp.time_gpt_model()
# sp.real_time_stock_market_data()
# sp.train_nbeats_model()
# # sp.stock_test_plot(end_date,new_date)
# sp.stock_future_plot(end_date,new_date)
