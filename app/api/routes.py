import shutil
import os
from fastapi import APIRouter, Depends, UploadFile, File
import asyncio
from langchain_chroma import Chroma
from app.agent.agent import build_agent, run_agent
from app.agent.graph import build_router_graph
from app.agent.supervisor import build_supervisor_graph
from app.agent.tools import get_last_chunks
from app.dependencies import get_vectordb
from app.rag.chunking import ingest_document
#from app.rag.evaluate import evaluate_rag
from app.rag.extraction import extract_financial_data
from app.rag.reranker import rerank
from app.rag.retriever import advanced_retrieval
from langchain_core.prompts import ChatPromptTemplate
from app.rag.memory import get_chat_history_as_string, get_or_create_memory, clear_memory, save_to_memory
from langfuse import observe, get_client
from langchain_groq import ChatGroq
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langfuse.langchain import CallbackHandler

try:
    from app.rag.evaluate import evaluate_rag
    RAGAS_ENABLED = True
except ImportError:
    RAGAS_ENABLED = False



router = APIRouter()
#VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
VECTORSTORE_DIR = "./app/vectorstore"
UPLOAD_DIR = "./data"


langfuse=get_client()

supervisor_graph = build_supervisor_graph()
supervisor_histories={}

def get_vectorstore():
    from langchain_cohere import CohereEmbeddings
    embedding_fn = CohereEmbeddings(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model="embed-english-v3.0"
    )
    return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embedding_fn)

def generate_answer(question:str, chunks:list, chat_history: str ="")->str:
    context="\n\n".join([ c.page_content for c in chunks])
    # llm = OllamaLLM(model="mistral",base_url=OLLAMA_BASE_URL)
    llm= ChatGroq(model="openai-gpt-oss-20b")
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
    result = chain.invoke({"context": context, "question": question, "chat_history": chat_history})
    return result.content if hasattr(result, 'content') else str(result)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...),company: str = "unknown", year: str = "unknown"):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
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

@router.post("/extract")
async def extract(question: str,company:str=None, year:str=None ):

    vectorstore=get_vectorstore()

    where_filter={}
    if company and year:
        where_filter={"$and": [{"company":company},{"year":year}]}
    elif company:
        where_filter = {"company": company}
    elif year:
        where_filter = {"year": year}
    
    chunks= advanced_retrieval(question,vectorstore,filters=where_filter if where_filter else None)
    reranked_chunks=rerank(question,chunks,top_k=6)

    

    context="\n\n".join([c.page_content for c in reranked_chunks])
    print("DEBUG CONTEXT SENT TO EXTRACTION:\n", context)
    extracted_data=extract_financial_data(context)

    return{
        "question":question,
        "filters_applied":{"company":company,"yesr":year},
        "extracted_data": extracted_data.model_dump(),
        "chunks_used":len(reranked_chunks)
    }



# PERF: advanced_retrieval uses Ollama/Mistral for MultiQuery + HyDE
# causing 58s latency. Fix: move to Groq llm for query generation.
# Identified via LangFuse trace aa42adcf on 2026-07-06.


@router.post("/chat/memory")
@observe()
async def chat(question: str, session_id: str = "default"):
   
    vectorstore = get_vectorstore()
    chat_history = get_chat_history_as_string(session_id)

    search_query = f"{chat_history}\n{question}" if chat_history else question

    with langfuse.start_as_current_observation(as_type="span",name="retrieval") as span:
      chunks = advanced_retrieval(search_query, vectorstore)
      span.update(output={"chunks_out":len(chunks)})

    with langfuse.start_as_current_observation(as_type="span",name="rerank") as span:
     reranked_chunks = rerank(question, chunks, top_k=3)
     span.update(output={"reranked_chunks":len(reranked_chunks)})
   
    with langfuse.start_as_current_observation(as_type="span",name="generate-rag-chat") as span:
      answer = generate_answer(question, reranked_chunks, chat_history=chat_history)
      span.update(output={"answer": answer})


    save_to_memory(session_id, question, answer)

    
    scores = {"faithfulness": None, "answer_relevancy": None}
    if RAGAS_ENABLED:
        scores = await asyncio.get_event_loop().run_in_executor(
            None, evaluate_rag, question, answer, reranked_chunks
        )
        if scores["faithfulness"] is not None:
            langfuse.score_current_trace(name="faithfulness", value=scores["faithfulness"], comment="RAGAS faithfulness")
        if scores["answer_relevancy"] is not None:
            langfuse.score_current_trace(name="answer_relevancy", value=scores["answer_relevancy"], comment="RAGAS answer relevancy")

    return {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "sources": [{"content": c.page_content[:200], "company": c.metadata.get("company", "unknown"), "year": c.metadata.get("year", "unknown"), "page": c.metadata.get("page", "unknown")} for c in reranked_chunks]
    }

@router.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    cleared = clear_memory(session_id)
    return {"cleared": cleared, "session_id": session_id}


@router.post("/agent/graph/ask")
async def agent_graph_ask(question:str,vectordb=Depends(get_vectordb)):
    graph = build_router_graph(vectordb)
    result = graph.invoke({"question": question})
    return {
        "question": result["question"],
        "question_type": result["question_type"],
        "answer": result["answer"],
        "chunks_used": len(result["chunks"])
    }

agent_instance = build_agent()

@router.post("/agent/ask")
@observe()
async def agent_ask(question: str, session_id: str = "default"):
    answer = run_agent(agent_instance, question, session_id=session_id)

    chunks = get_last_chunks()
    scores = {}

    scores = {"faithfulness": None, "answer_relevancy": None}
    if chunks and RAGAS_ENABLED:
        scores = await asyncio.get_event_loop().run_in_executor(
            None, evaluate_rag, question, answer, chunks
        )
        if scores["faithfulness"] is not None:
            langfuse.score_current_trace(name="faithfulness", value=scores["faithfulness"], comment="RAGAS faithfulness - agent")
        if scores["answer_relevancy"] is not None:
            langfuse.score_current_trace(name="answer_relevancy", value=scores["answer_relevancy"], comment="RAGAS answer relevancy - agent")

    return {
        "question": question,
        "answer": answer,
        "chunks_used": len(chunks),
        "scores": {
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy")
        }
    }


@router.post("/agent/graph/supervisor")
@observe()
async def supervisor_ask(question:str,session_id:str="default"):

    handler=CallbackHandler()
    history=supervisor_histories.get(session_id, [])

    try:
     result= supervisor_graph.invoke({
        "question":question,
        "plan":None,
        "agents_to_call": None,
        "retrieval_output":None,
        "calculation_output":None,
        "market_output":None,
        "final_answer":None,
        "retrieval_chunks":None,
        "iterations":0,
        "max_iterations": 3,
        "history":history
     },
     config={"callbacks": [handler]})

     supervisor_histories[session_id] = result.get("history", [])

     return {
        "question":question,
        "answer": result["final_answer"],
        "iterations":result["iterations"],
        "plan": result["plan"],
        "session_id":session_id
     }
    except Exception as e:
       raise
    finally:
       langfuse.flush