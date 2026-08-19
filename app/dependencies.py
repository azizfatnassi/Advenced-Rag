from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
import os

VECTORSTORE_DIR = "./app/vectorstore"

def get_vectordb():
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="embed-english-v3.0"
        )
    )