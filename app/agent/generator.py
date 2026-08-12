

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from app.agent.llm import groq_llm

load_dotenv()

FACTUAL_PROMPT= ChatPromptTemplate.from_template(
""" You are a financial analyst. Answer the question directly 
and concisely.
State the exact number or fact from the context.
If the answer is not in the context, say "Not found in the provided documents."

Context: {context}
Question: {question}
Answer: """)

COMPARATIVE_PROMPT =ChatPromptTemplate.from_template("""

 You are a financial analyst. Compare the data across the time periods or 
companies mentioned in the question. Use exact numbers from the context.
Structure your answer clearly with the comparison.

Context: {context}
Question: {question}
Answer:
""")

ANALYTICAL_PROMPT = ChatPromptTemplate.from_template("""
You are a senior financial analyst. Think through this step by step.
First identify the relevant numbers from the context.
Then explain the reasoning behind the trend or pattern.
Support your analysis with specific figures.

Context: {context}
Question: {question}
Analysis:
""")

PROMPTS= {
    "factual":FACTUAL_PROMPT,
    "comparative":COMPARATIVE_PROMPT,
    "analytical":ANALYTICAL_PROMPT
}

def generate_answer(question:str,chunks:list,question_type:str)->str:
    context="\n\n".join([c.page_content for c in chunks])
    prompt=PROMPTS.get(question_type,FACTUAL_PROMPT)
    chain= prompt | groq_llm
    return chain.invoke({"context":context,"question":question}).content

