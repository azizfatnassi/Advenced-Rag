from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

VECTORSTORE_DIR = "./app/vectorstore"

def get_vectordb():
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    )