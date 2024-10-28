import os
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
import ssl
import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)
class CustomSession(requests.Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verify = False

http_client = httpx.Client(verify=False)
https_client=CustomSession()
load_dotenv()
http_client = httpx.Client(verify=False)

def ai_agent(name, media_type) :
    llm=ChatOpenAI(temperature=0)
    summary_template = """
    Please give me the 10 latest news pertaining to {name_of_person} company . Don't give me the values
    Get me those as the output.
    """
    search = SearchApiAPIWrapper(http_client=https_client)
    summary_prompt_template = PromptTemplate(input_variables=["name_of_person"], template=summary_template)
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
    print("The name is",name)
    result=agent_executor.invoke(
        input={"input":summary_prompt_template.format_prompt(name_of_person=name)}
    )
    linkedln_url=result
    with open(os.getcwd()+"/ouptut.txt","w") as file:
        file.write(str(linkedln_url))
    return linkedln_url

def real_time_stock_news(name,media_type):
    ssl._create_default_https_context = ssl._create_unverified_context
    linkedln_username=ai_agent(name=name,media_type=media_type)
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

if __name__=="__main__":
    load_dotenv()
    abs_path=os.getcwd()
    print(abs_path)
    real_time_stock_news("Apple company","nothing")
    

   