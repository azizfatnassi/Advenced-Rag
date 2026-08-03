from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os


VECTORSTORE_DIR= "./app/vectorstore"
OLLAMA_BASE_URL= os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
from langchain_community.embeddings import SentenceTransformerEmbeddings

def get_vectordb():

    return Chroma(persist_directory=VECTORSTORE_DIR,
            embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")     
                  )