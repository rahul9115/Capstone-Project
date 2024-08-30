import os
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import httpx
from linkedln_lookup_agent import lookup as linkedln_lookup_agent
# from agents.tools import get_profile_url_tavily
import ssl
import pandas as pd
import csv
import io
http_client = httpx.Client(verify=False)
def ice_break_with(name,media_type):
    ssl._create_default_https_context = ssl._create_unverified_context
    linkedln_username=linkedln_lookup_agent(name=name,media_type=media_type)
    print("The username is :",linkedln_username)
    # linkedln_data=scrape_linkedln_profile(linkedln_profile_url=linkedln_username)
    open_ai_key=os.environ['OPENAI_API_KEY']
    summary_template="""
    Based on the given information
    Information: {information} 
    give me the sentiment of the news
    """
    summary_prompt_template= PromptTemplate(template=summary_template)
    llm=ChatOpenAI(temperature=0,http_client=http_client)
    chain=summary_prompt_template|llm 
    res=chain.invoke(input={"information":linkedln_username})
    print("LLM output",res)
    # csv_data = res.split("csv")[1].split("")[0].strip()
    # csv_filename = "output.csv"
    # print("The result",res)

if __name__=="__main__":
    print("Langchain")
    load_dotenv()
    abs_path=os.getcwd()
    print(abs_path)
    ice_break_with("Apple company","nothing")
    # df=pd.read_excel(abs_path+"\\ice_breaker\\input.xlsx")
    # df=df[2:]
   
    # lf=[]
    # tf=[]
    # ff=[]
    # c=[]
    # print(df.columns)
    # for i in range(len(df)):
    #     try:
    #         company=df.iloc[i].values[0]
    #         c.append(company)code 
    #         for j in ["Linkedln","Tiktok","Facebook"]:
    #             try:
    #                 res=ice_break_with(f"{company} Company {j}",j)
    #                 if j=="Linkedln":
    #                     lf.append(res.split(" ")[0])
    #                 if j=="Tiktok":
    #                     tf.append(res.split(" ")[0])
    #                 if j=="Facebook":
    #                     ff.append(res.split(" ")[0])
    #             except:
    #                 continue
    #     except:
    #         continue
        
            
    #     data={"name":c,"Linkedln Followers":lf,"Tiktok Followers":tf,"Facebook Followers":ff}
    #     new_df=pd.DataFrame(data)
    #     new_df.to_csv(abs_path+"\\output.csv")
    

   