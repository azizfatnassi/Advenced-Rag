from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)