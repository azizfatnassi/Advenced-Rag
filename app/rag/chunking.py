

import os 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma 

VECTORSTORE_DIR = "./vectorstore"


def load_file(file_path:str):
    "load a file"
    ext= os.path.splitext(file_path)[-1].lower()
    
    if ext ==".pdf":
        loader= PyPDFLoader(file_path)


    elif ext ==".txt":
        loader= TextLoader(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load() 

def ingest_document(file_path: str):
    
    documents= load_file(file_path)

    splitter= RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap= 80 
    )
    
    chunks= splitter.split_documents(documents)

    print(f'split into {len(chunks)} chunks')

    embedding_fn= OllamaEmbeddings(model="nomic-embed-text")

    vectorstore= Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embedding_fn
    )

    existing= vectorstore.get()
    if existing["ids"]:
        vectorstore.delete(ids=existing["ids"])
        print(f"🗑️  Cleared {len(existing['ids'])} old chunks")

    vectorstore.add_documents(chunks)
    print(f"✅ Done. {len(chunks)} chunks saved to vectorstore.")
    
    return vectorstore

if __name__ == "__main__":
    ingest_document("../data/roadmap-text.txt")