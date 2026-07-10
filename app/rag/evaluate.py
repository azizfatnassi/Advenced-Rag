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
print("API KEY LOADED:", os.getenv("GROQ_API_KEY") is not None)
print("API KEY VALUE:", os.getenv("GROQ_API_KEY")[:10] if os.getenv("GROQ_API_KEY") else "NONE")

answer_relevancy = AnswerRelevancy(strictness=1)

def evaluate_rag(question: str, answer: str, contexts: list) -> dict:
    llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=2000
    ))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    samples = [
        SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=[c.page_content[:500] for c in contexts],
        )
    ]

    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    print("RAW RESULT:", result)
    print("RAW SCORES:", result.scores)

    def safe_score(val):
        try:
            v = float(val[0]) if isinstance(val, list) else float(val)
            return round(v, 3) if not math.isnan(v) and not math.isinf(v) else None
        except:
            return None

    return {
        "faithfulness": safe_score(result["faithfulness"]),
        "answer_relevancy": safe_score(result["answer_relevancy"]),
    }