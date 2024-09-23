from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
load_dotenv()
if __name__=="__main__":
    embeddings=OpenAIEmbeddings()
    llm=ChatOpenAI()
    query="What is pinecone in machine learning"
    chain=PromptTemplate.from_template(template=query) | llm
    result=chain.invoke(input={})
    vectorstore=PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)
    retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain=create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
    retrieval_chain=create_retrieval_chain(
        retriever=vectorstore.as_retriever(),combine_docs_chain=combine_docs_chain
    )
    result=retrieval_chain.invoke(input={"input":query})
    print(result)

