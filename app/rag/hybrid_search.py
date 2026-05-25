
from rank_bm25 import BM25Okapi



def bm25_search(question: str, chunks:list,top_k: int=5)->list:

    if not chunks:
        return[]
    tokenized_chunks= [chunk.page_content.lower().split() for chunk in chunks]

    bm25=BM25Okapi(tokenized_chunks)

    tokenized_question= question.split()
    scores=bm25.get_scores(tokenized_question)

    scored_chunks= sorted(zip(scores,chunks), key=lambda x:x[0],reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]