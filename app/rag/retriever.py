from app.rag.hybrid_search import bm25_search
from app.rag.multiquery import generate_queries

def advanced_retrieval(question: str, vectordb, k: int = 5,filters: dict = None) -> list:
    queries = generate_queries(question)
    
    all_chunks = []
    seen_ids = set()
    
    for query in queries:  # ✅ loop through each query
        if filters:

            results = vectordb.similarity_search(query, k=k, filter=filters)
        else :   
            results = vectordb.similarity_search(query, k=k)  # ✅ one string at a time
        
        for chunk in results:
            chunk_id = hash(chunk.page_content)  # ✅ hash the text content
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_chunks.append(chunk)


    # re-score and combine 

    bm25_results= bm25_search(question,all_chunks,top_k=10)

    final_chunks=[]
    final_seen=set()

    for chunk in all_chunks + bm25_results:
        chunk_id=hash(chunk.page_content)
        if chunk_id not in final_seen:
            final_seen.add(chunk_id)
            final_chunks.append(chunk)
    
    return final_chunks[:20]