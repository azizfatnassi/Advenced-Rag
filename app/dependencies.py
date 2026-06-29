from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings



VECTORSTORE_DIR= "./app/vectorstore"


def get_vectordb():

    return Chroma(persist_directory=VECTORSTORE_DIR,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")      
                  )