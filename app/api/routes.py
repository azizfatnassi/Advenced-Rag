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
from app.rag.memory import get_chat_history_as_string, get_or_create_memory, clear_memory, save_to_memory




router = APIRouter()
import os
#VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
VECTORSTORE_DIR = "./app/vectorstore"
UPLOAD_DIR = "./data"

def get_vectorstore():
    embedding_fn = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embedding_fn)


def generate_answer(question:str, chunks:list, chat_history: str ="")->str:
    context="\n\n".join([ c.page_content for c in chunks])
    llm = OllamaLLM(model="mistral")
    prompt=ChatPromptTemplate.from_template(
        """""You are a helpful assistant. Use the context 
             and conversation history to answer"


 Conversation history:
 {chat_history}  


 Context:
 {context}

 Question: {question}

 Answer:
 """)
    chain= prompt | llm
    return chain.invoke({"context": context, "question": question, "chat_history": chat_history})
    

@router.post("/upload")
async def upload_file(file: UploadFile = File(...),company: str = "unknown", year: str = "unknown"):
    # Save file to data/
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Ingest it
    ingest_document(file_path,company=company,year=year)
    
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

@router.post("/ask/filetred")
async def ask_filtred(question:str,company:str=None,year:str=None):
    vectorstore=get_vectorstore()

    where_filter={}
    if company and year :
        where_filter= {"$and":[{"company":company},{"year":year}]}
    elif company:
        where_filter = {"company": company}
    elif year:
        where_filter = {"year": year}
    chunks = advanced_retrieval(question, vectorstore, filters=where_filter if where_filter else None)
    reranked_chunks = rerank(question, chunks, top_k=3)
    answer = generate_answer(question, reranked_chunks)
    
    return {
        "question": question,
        "filters_applied": {"company": company, "year": year},
        "answer": answer,
        "chunks_found": len(chunks)
    }
@router.post("/chat/memory")
async def chat(question: str, session_id: str = "default"):
    vectorstore = get_vectorstore()

  
    chat_history = get_chat_history_as_string(session_id)
    search_query= f"{chat_history}\n{question}" if chat_history else question
    chunks = advanced_retrieval(search_query, vectorstore)
    reranked_chunks = rerank(question, chunks, top_k=3)
    answer = generate_answer(question, reranked_chunks, chat_history=chat_history)

   
    save_to_memory(session_id, question, answer)

    return {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "sources": [{
            "content": c.page_content[:200],
            "company": c.metadata.get("company","unknown"),
            "year":c.metadata.get("year","unknown"),
            "page":c.metadata.get("page","unknown")
        }
        for c in reranked_chunks
        ]
    }

@router.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    cleared = clear_memory(session_id)
    return {"cleared": cleared, "session_id": session_id}