"""
Isolated RAGAS debug script.
Tests answer_relevancy on ONE hardcoded example (no Mistral, no retrieval, no 20-min wait)
to see the raw error/output instead of a silent NaN.
"""
import os
import math
from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Turn on ragas debug logging so we see WHY relevancy fails instead of just getting NaN
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("ragas").setLevel(logging.DEBUG)

answer_relevancy = AnswerRelevancy(strictness=1)

llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

sample = SingleTurnSample(
    user_input="What was Tesla's total revenue in 2023?",
    response="The total revenue for Tesla, Inc. in 2023 was $96,773 million.",
    retrieved_contexts=["Tesla's total revenue in fiscal year 2023 was $96,773 million, an increase from the prior year."],
)

dataset = EvaluationDataset(samples=[sample])

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings,
    raise_exceptions=True,  # <-- KEY CHANGE: force it to crash loudly instead of swallowing the error as NaN
)

print("\n\n=== FINAL RESULT ===")
print(result)
print(result.scores)