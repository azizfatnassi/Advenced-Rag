
#from sentence_transformers import CrossEncoder
#import os

#os.environ["TRANSFORMERS_OFFLINE"]="1"
#os.environ["HF_DATASETS_OFFLINE"]="1"
#_model=None

#def get_model():
 # global _model
  #if _model is None:
  #  _model =CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
  #return _model

#model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#def rerank(question:str, chunks: list, top_k=3)->list:
 # model= get_model()
 # pairs=[[question,chunk.page_content] for chunk in chunks]
 # scores=model.predict(pairs)

 # scored_chunks=sorted(zip(scores,chunks),key=lambda x:x[0],reverse=True)
  #return [chunk for _, chunk in scored_chunks[:top_k]]



import cohere
import os

_client = None

def get_client():
    global _client
    if _client is None:
        _client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
    return _client

def rerank(question: str, chunks: list, top_k=3) -> list:
    client = get_client()
    
    # Extract text from chunks (same as before, chunks are LangChain Documents)
    texts = [chunk.page_content for chunk in chunks]
    
    if not texts:
        return chunks
    
    results = client.rerank(
        model="rerank-v3.5",
        query=question,
        documents=texts,
        top_n=top_k
    )
    
    # Return the original chunk objects in reranked order
    return [chunks[r.index] for r in results.results]

