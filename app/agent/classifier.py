
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.agent.llm import groq_llm

load_dotenv()


CLASSIFY_PROMPT= ChatPromptTemplate.from_template("""
     you are a question classifier for a financial QA system
     Classify the question into exactly one of these types:
- factual: asks for a single specific fact, number, or date
- comparative: asks to compare, contrast, or show trends across time or companies
- analytical: asks for reasoning, explanation, or insight about financial data

Question: {question}

Reply with ONLY one word: factual, comparative, or analytical.
""")

def classify_question(question:str)->str:
    chain= CLASSIFY_PROMPT | groq_llm
    result= chain.invoke({"question":question})
    question_type= result.content.strip().lower()

    if question_type not in ["factual","com parative","analytical"]:
        return "factual"
    
    return question_type 

