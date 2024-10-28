from datasets import load_dataset
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import pinecone
import requests
import json
import csv
import pandas as pd
import pdfplumber
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
from langchain.chains import LLMChain
import warnings
ds = load_dataset("rohith2812/atoigeneration_sample")
print(ds)
# exit()
warnings.filterwarnings("ignore")
# load_dotenv()
class JobSearch:
    # def jobs_api_call(self):
    #     API_credentials = (os.environ.get("OXYLAB_USERNAME"), os.environ.get("OXYLAB_PASSWORD"))
    #     payload = {
    #         'source': 'universal',
    #         'url': 'https://stackshare.io/jobs',
    #         'geo_location': 'United States',
    #         'render': 'html',
    #         'browser_instructions': [
    #             {
    #                 'type': 'click',
    #                 'selector': {
    #                     'type': 'xpath',
    #                     'value': '//button[contains(text(), "Load more")]'
    #                 }
    #             },
    #             {'type': 'wait', 'wait_time_s': 2}
    #         ] * 13 + [
    #             {
    #                 "type": "fetch_resource",
    #                 "filter": "^(?=.*https://km8652f2eg-dsn.algolia.net/1/indexes/Jobs_production/query).*"
    #             }
    #         ]
    #     }
    #     response = requests.request(
    #         'POST',
    #         'https://realtime.oxylabs.io/v1/queries',
    #         auth=API_credentials, 
    #         json=payload, 
    #         timeout=180
    #     )
    #     print("Response",response.json())
    #     results = response.json()['results'][0]['content']
    #     print(results)
    #     data = json.loads(results)

    #     jobs = []
    #     for job in data['hits']:
    #         parsed_job = {
    #             'Title': job.get('title', ''),
    #             'Location': job.get('location', ''),
    #             'Remote': job.get('remote', ''),
    #             'Company name': job.get('company_name', ''),
    #             'Company website': job.get('company_website', ''),
    #             'Verified': job.get('company_verified', ''),
    #             'Apply URL': job.get('apply_url', '')
    #         }
    #         jobs.append(parsed_job)

    #     fieldnames = [key for key in jobs[0].keys()]
    #     with open('stackshare_jobs.csv', 'w') as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for item in jobs:
    #             writer.writerow(item)
        

    def store_data_pinecone(self,index_name,pdf=False):
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        if pdf:
            with pdfplumber.open(os.getcwd()+"/Vemuri Rahul.pdf") as pdf:
                first_page = pdf.pages[0]
                text = first_page.extract_text()
                print("text",text)

            print("Splitting.. ")
            documents = [Document(page_content=text)]
            text=TextLoader(text)
            text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
            texts=text_splitter.split_documents(documents)
            # print(f"Created {len(text_splitter)} chunks")

            print("Ingesting PDF")
            embeddings=OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
            PineconeVectorStore.from_documents(texts,embeddings,index_name=index_name)
        else:
            csv_loader = CSVLoader(file_path=f"{os.getcwd()}\stackshare_jobs.csv")
            documents = csv_loader.load()
            embeddings=OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
            vectorstore = Pinecone(index_name=index_name, embedding_function=embeddings)

            print("Ingesting CSV")
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            PineconeVectorStore.from_documents(documents,embeddings,index_name=index_name)
    def RAG(self):
        print("In here")
        embeddings=OpenAIEmbeddings()
        
        os.environ["OPENAI_API_KEY"] = "sk-M6KVmE1xnvhTRvX5Opis5O53CJ2L8apJe3nAgHJsXJT3BlbkFJa3fq9qIQMqs2O1aLUueBEDsm4CPRiCosZlvlM2L5IA"
        print("This",os.environ.get("OPENAI_API_KEY"))
        llm=ChatOpenAI()
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        query="What data is in the index explain it to me"
        chain=PromptTemplate.from_template(template=query) | llm
        # vectorstore=PineconeVectorStore(index_name="resume-index",embedd  ing=embeddings)
        vectorstore=PineconeVectorStore(index_name="atoigeneration",embedding=embeddings)
        retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain=create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
        retrieval_chain=create_retrieval_chain(
            retriever=vectorstore.as_retriever(),combine_docs_chain=combine_docs_chain
        )

        resume_summary=retrieval_chain.invoke(input={"input":query})
        print("This",resume_summary)
        exit()
        query="Based on my resume summary {resume_summary}  give me relevant jobs based on job description. The final answer should be the Apply URL"
        job_prompt=PromptTemplate(input_variables=["resume_summary"],template=query)
        job_chain = LLMChain(prompt=job_prompt, llm=llm)
        job_query = job_prompt.format(resume_summary=resume_summary['context'])
        job_query=job_prompt.format(resume_summary=resume_summary["context"][0])
        result = job_chain.run(job_query)
        vectorstore=PineconeVectorStore(index_name="jobs-index",embedding=embeddings)
        retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain=create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
        retrieval_chain=create_retrieval_chain(
            retriever=vectorstore.as_retriever(),combine_docs_chain=combine_docs_chain
        )
        result=retrieval_chain.invoke(input={"input":result})
        print("Result", result)

search=JobSearch()
# search.jobs_api_call()
# search.store_data_pinecone("resume-index",pdf=True)
# search.store_data_pinecone("jobs-index",pdf=False)
# search.store_data_pinecone("atoi",pdf=False)
search.RAG()
