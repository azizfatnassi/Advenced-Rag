
from sentence_transformers import CrossEncoder
import os

os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_DATASETS_OFFLINE"]="1"
_model=None

def get_model():
  global _model
  if _model is None:
    _model =CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
  return _model

#model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question:str, chunks: list, top_k=3)->list:
  model= get_model()
  pairs=[[question,chunk.page_content] for chunk in chunks]
  scores=model.predict(pairs)

  scored_chunks=sorted(zip(scores,chunks),key=lambda x:x[0],reverse=True)
  return [chunk for _, chunk in scored_chunks[:top_k]]

