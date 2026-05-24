
from sentence_transformers import CrossEncoder

model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question:str, chunks: list, top_k=3)->list:
  pairs=[[question,chunk.page_content] for chunk in chunks]
  scores=model.predict(pairs)

  scored_chunks=sorted(zip(scores,chunks),key=lambda x:x[0],reverse=True)
  return [chunk for _, chunk in scored_chunks[:top_k]]

