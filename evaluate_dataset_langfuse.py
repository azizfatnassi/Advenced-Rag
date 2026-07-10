# evaluate_dataset_langfuse.py
import os
import time
from dotenv import load_dotenv
load_dotenv()

from langfuse import get_client
from app.agent.agent import build_agent, run_agent
from app.rag.evaluate import evaluate_rag
from app.agent.tools import get_last_chunks
from app.dependencies import get_vectordb
from app.rag.eval_dataset import EVAL_QUESTIONS

langfuse = get_client()
DATASET_NAME = "finance-rag-eval-v1"


def create_dataset():
    try:
        langfuse.get_dataset(DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' already exists — skipping creation")
        return
    except Exception:
        langfuse.create_dataset(name=DATASET_NAME)
        print(f"Created dataset '{DATASET_NAME}'")
        for item in EVAL_QUESTIONS:
            langfuse.create_dataset_item(
                dataset_name=DATASET_NAME,
                input={"question": item["question"]},
                expected_output={"answer": item["expected"]}
            )
        print(f"Added {len(EVAL_QUESTIONS)} items to dataset")


def make_task(agent):
    def task(*, item, **kwargs):
        question = item.input["question"]
        print(f"  Q: {question}")

        answer = run_agent(agent, question)
        chunks = get_last_chunks()

        if chunks:
            scores = evaluate_rag(question, answer, chunks)
            if scores["faithfulness"] is not None:
                langfuse.score_current_trace(
                    name="faithfulness",
                    value=scores["faithfulness"],
                )
            if scores["answer_relevancy"] is not None:
                langfuse.score_current_trace(
                    name="answer_relevancy",
                    value=scores["answer_relevancy"],
                )
            print(f"    faithfulness={scores['faithfulness']}, relevancy={scores['answer_relevancy']}")
        else:
            print(f"    no chunks — skipping RAGAS")

        print(f"    answer: {answer[:80]}...")
        print(f"    waiting 15s for rate limit...")
        time.sleep(15)

        return answer

    return task


def run_experiment(experiment_name: str):
    print(f"\nRunning experiment: {experiment_name}")
    agent = build_agent()
    dataset = langfuse.get_dataset(DATASET_NAME)

    dataset.run_experiment(
        name=experiment_name,
        task=make_task(agent),
        max_concurrency=1,
        metadata={
            "change": "improved system propmt- added explicit calculate rule for margins and retry rule for failed search",
            "hypothesis":"agent fails on derived metrics because prompt doesnt tell it to calculate from raw numbers ",
            "models":"qwen/qwen3.6-27b agent,llama-3.1-8b-instant evaluation"
        }
    )

    langfuse.flush()
    print(f"\nDone. Check LangFuse → Datasets → {DATASET_NAME} → Runs tab")


if __name__ == "__main__":
    create_dataset()
    run_experiment("experiment-2-better-prompt")