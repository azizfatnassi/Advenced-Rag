# advanced/multi_query.py
from langchain_core.prompts import ChatPromptTemplate 
from langchain_groq import  ChatGroq

llm = ChatGroq(model="openai-gpt-oss-20b",temperature=0)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template("""
You are an AI assistant. Your task is to generate 3 different 
versions of the user's question to improve document retrieval.

Generate 3 variations that:
- Use different words but same meaning
- Approach the question from different angles
- Are specific and searchable

Original question: {question}

Output ONLY the 3 questions, one per line, no numbering.
""")

def generate_queries(question: str) -> list[str]:
    chain = MULTI_QUERY_PROMPT | llm
    result = chain.invoke({"question": question})
    
    # Parse the 3 questions
    queries = [q.strip() for q in result.content.strip().split("\n") if q.strip()]
    
    # Always include original
    queries.append(question)
    
    return queries[:4]  # max 4 queries including original