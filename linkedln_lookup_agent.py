import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub
import httpx
# from agents.tools import get_profile_url_tavily
from langchain_community.utilities import SearchApiAPIWrapper
# from tools import get_profile_url_tavily
import ssl
import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

# Create a custom session class
class CustomSession(requests.Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verify = False

http_client = httpx.Client(verify=False)
https_client=CustomSession()
load_dotenv()

def lookup(name, media_type) :
    llm=ChatOpenAI(temperature=0)
    summary_template = """
    Please search for job roles related to  {name_of_person} on Linkedln. 
    Get me those as the output.
    """
    search = SearchApiAPIWrapper(http_client=https_client)
    summary_prompt_template = PromptTemplate(input_variables=["name_of_person"], template=summary_template)
    open_ai_key = os.environ['OPENAI_API_KEY']
   
    tools_for_agent=[
        Tool(
            name=f"Search for Jobs",
            func=search.run,
            description=f"Useful for getting available jobs from search APIs"
        )
    ]
    # ssl._create_default_https_context = ssl._create_unverified_context
    requests.Session.verify = False
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


