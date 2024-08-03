from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
if __name__=="__main__":
    load_dotenv()
    loader=TextLoader("C:/Users/sudha/OneDrive/Documents/RAG/medium.txt",encoding="utf-8")
    document=loader.load()
    
    print("Splitting.. ")
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts=text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings=OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting")
    PineconeVectorStore.from_documents(texts,embeddings,index_name=os.environ["INDEX_NAME"])