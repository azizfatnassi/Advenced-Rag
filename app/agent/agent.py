
import os

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

# langfuse pinned to 2.60.0 — newer versions removed .trace() method
# which breaks Phase 2 /chat/memory endpoint
# do not upgrade until Phase 2 LangFuse code is migrated to new SDK API

from app.agent.tools import calculate, get_stock_price, search_documents
#create_react_agent is depricated in new versions of langraph
#migrate to create_agent from langchain.agents when upgrading/updating langchain
from langgraph.prebuilt import create_react_agent 
from app.agent.llm import groq_llm
#from langfuse.callback import CallbackHandler
from app.agent.tools import set_trace



langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)

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

def run_agent(agent,question:str,user_id:str= "dev", question_type: str = "unknown")->str:
  
  trace= langfuse.trace(name="agent-run",
        user_id= user_id,
        metadata={
            "question_type": question_type,
            "model":"qwen/qwen3.6-27b",
                  },
        input=question
        )
  set_trace(trace)
  try:
      span= trace.span(name="agent-invoke",input=question)

      result= agent.invoke(
          {"messages": [{"role":"user","content":question}]}
      )
      answer=result["messages"][-1].content
      span.end(output=answer)
      trace.update(output=answer)

      return answer
  except Exception as e:
      trace.update(output=f"ERROR:{str(e)}")
      raise
  finally :
     set_trace(None)
     langfuse.flush()

     



if __name__ == "__main__":
    agent = build_agent()

    test_questions = [
        "What was Tesla's revenue in 2023?",
        "What is Tesla's current stock price?",
        "Calculate Tesla's revenue growth from 2021 to 2023",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {q}")
        answer = run_agent(agent, q)
        print(f"ANSWER: {answer}")