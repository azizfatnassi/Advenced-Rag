
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

llm=ChatGroq(model="openai-gpt-oss-20b",temperature=0)

HYDE_PROMPT=ChatPromptTemplate.from_template(""" Write a short factual
       paragraph that would answer the question asked .
       write it as if you found it in a financial document .
      Do not say " i think " or "maybe" write it as a fact 

    Question: {question}
    Paragraph: """)


def hyde_answer(question: str)->str:
    chain= HYDE_PROMPT | llm 
    return chain.invoke({"question":question}).content