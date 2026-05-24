import shutil
import os
from fastapi import APIRouter, UploadFile, File
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from app.rag.chunking import ingest_document
from app.rag.evaluate import evaluate_rag
from app.rag.reranker import rerank
from app.rag.retriever import advanced_retrieval
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
router = APIRouter()

VECTORSTORE_DIR = "./app/vectorstore"
UPLOAD_DIR = "./data"

def get_vectorstore():
    embedding_fn = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embedding_fn)


def generate_answer(question:str, chunks:list):
    context="\n\n".join([ c.page_content for c in chunks])
    llm = OllamaLLM(model="mistral")
    prompt=ChatPromptTemplate.from_template("""""Use the following context to answer the question clearly and structured.
 If the answer is not in the context, say "I don't have enough information."

 Context:
 {context}

 Question: {question}

 Answer:
 """)
    chain= prompt | llm
    return chain.invoke({"context": context, "question": question})
    

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save file to data/
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Ingest it
    ingest_document(file_path)
    
    return {"message": f"{file.filename} uploaded and ingested successfully"}

@router.post("/ask")
async def ask(question: str):
    vectorstore = get_vectorstore()
    chunks = vectorstore.similarity_search(question, k=5)
    answer = generate_answer(question, chunks)
    return {"question": question, "answer": answer}

@router.post("/ask/advanced")
async def ask_advanced(question: str):
    vectorstore= get_vectorstore()
    chunks= advanced_retrieval(question,vectorstore,k=5)
    reranked_chunks = rerank(question, chunks, top_k=3)
    
    answer = generate_answer(question, chunks)
    return {
        "question": question,
        "answer": answer,
        "chunks_before_rerank": len(chunks),
        "chunks_after_rerank": len(reranked_chunks)
    }

@router.post("/evaluate")
async def evaluate(question: str):
    vectorstore = get_vectorstore()
    chunks = advanced_retrieval(question, vectorstore)
    reranked_chunks = rerank(question, chunks, top_k=3)
    answer = generate_answer(question, reranked_chunks)
    scores = evaluate_rag(question, answer, reranked_chunks)
    
    return {
        "question": question,
        "answer": answer,
        "scores": scores
    }