from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Hello World")
    loader = TextLoader("D:\VSC Projects\intro_vector_db\mediumblog1.txt") 
    document = loader.load()

    print("splitting the document into chunks")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)
    print(chunks)

    print("Creating embeddings")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting data")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.environ.get("INDEX_NAME"))

    print("Data ingestion complete")
    
