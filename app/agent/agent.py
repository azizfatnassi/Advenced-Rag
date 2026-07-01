



from app.agent.tools import calculate, get_stock_price, search_documents
from langgraph.prebuilt import create_react_agent 
from app.agent.llm import groq_llm


SYSTEM_PROMPT= """ You are a financial expert and assisntant with access to 
a datbase of company financial documents and treal time market data 
you have access to these tools:
-search_documents: search financial reports for specific data 
-calculate: perform math on financial numbers
-get_stock_price: get current stock price for a ticker

Rules: 
1- Always search data before trying to answer from memory
2-Use calculate when you need to compute growth rates, ratios or percentages
3-Be specifique - cite the source company and year whene referencing numbers
4- If data is not found, say so clearly — never hallucinate numbers"""


def build_agent():
    tools=[search_documents, calculate, get_stock_price]
    agent=create_react_agent(
        model=groq_llm,
        tools=tools,
        prompt=SYSTEM_PROMPT

    )
    return agent

if __name__== "__main__":
    agent= build_agent()

    test_questions = [
        "What was Tesla's revenue in 2023?",
        "What is Tesla's current stock price?",
        "Calculate Tesla's revenue growth from 2021 to 2023",
    ]

    for q in test_questions: 
        print(f"\n{'='*60}")
        print(f"QUESTION: {q}")
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        print(f"ANSWER: {result['messages'][-1].content}")