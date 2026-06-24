# Advanced RAG Finance

## What it does

A conversational AI assistant for financial documents. Upload any financial PDF (annual reports, earnings, filings) and ask questions in plain English. The system remembers your conversation so you can ask follow-up questions naturally without repeating context.

Built for analysts, investors, or anyone who needs to pull insights out of financial documents without reading hundreds of pages manually.

On top of Q&A, there's also a structured extraction endpoint that pulls financial metrics (revenue, net income, R&D spend, etc.) out of retrieved context into clean, validated JSON instead of a paragraph — meant for cases where you want the numbers in a database or a downstream system rather than as a sentence.

## Architecture

```
User → Streamlit UI → FastAPI Backend → Retrieval Pipeline → Mistral 7B → Answer
                              ↓
                    MultiQuery (3 variations)
                    HyDE (hypothetical answer)
                    BM25 (keyword search)
                    Merge all results
                    CrossEncoder Reranking
                    Memory injection
                    Mistral generates answer
```

There's a second path off the same retrieval pipeline: instead of generating a free-text answer, the reranked chunks get handed to Instructor + Groq, which returns structured JSON matching a fixed schema. Same retrieval, two different outputs depending on what you need.

## Techniques used and why

**MultiQuery Retrieval** — One question = one angle = missed chunks. Mistral generates 3 variations of your question to cover more semantic ground.

**HyDE (Hypothetical Document Embeddings)** — Questions and answers live in different vector spaces. We ask Mistral to write a hypothetical answer first, then search with that. Answer-to-answer similarity is more accurate than question-to-answer.

**BM25 Hybrid Search** — Semantic search understands meaning but misses exact terms. BM25 catches exact matches like ticker names, figures, and quarter labels. Finance documents need both.

**CrossEncoder Reranking** — Similarity search retrieves candidates fast but loosely. The cross-encoder reads question and chunk together to score actual relevance. Slower but a lot more accurate on the final selection. Q&A endpoints use `top_k=3`; the extraction endpoint uses a wider `top_k=6` since pulling multiple metrics (revenue, net income, R&D, etc.) at once means the relevant numbers are often spread across different sections of the document, not sitting in one place.

**Conversation-Aware Retrieval** — Follow-up questions like "how does that compare?" are meaningless without context. Chat history gets combined with the current question before searching ChromaDB.

**Session Memory** — Each conversation has a session ID. History is stored and injected into every prompt so Mistral maintains context across turns.

**Structured Extraction (Instructor)** — Free-text answers can't be consumed by other systems without re-parsing them, which is fragile. The `/extract` endpoint wraps a Groq call (`llama-3.3-70b-versatile`) with Instructor in `Mode.TOOLS`, using constrained decoding so the model's output is mechanically restricted to match a Pydantic schema — it's not just "asked nicely" to return JSON, it structurally can't return anything else. Every numeric field in the schema is `Optional`, so if a value isn't in the retrieved context, it comes back as `None` instead of the model inventing something plausible-sounding. There's also a `not_found_in_context` flag to separate "this chunk has no financial data" from an actual extraction failure. Tested against both a clean positive case and a context with zero financial data, and both behaved correctly.

**Observability (LangFuse)** — Every request through `/chat/memory` is traced end to end — retrieval, reranking, and generation each get their own span with timing and input/output captured. Useful for figuring out *where* a bad answer actually went wrong instead of just staring at the final output. Runs locally via Docker (`langfuse:2`) with a Postgres backend, dashboard at `localhost:3000`.

## Benchmark results

Evaluated on 10 financial questions from Tesla's 2023 Annual Report, using RAGAS with Groq (`llama-3.3-70b-versatile`) as the judge model.

| Metric | Score |
|---|---|
| Faithfulness | 0.802 / 1.0 |
| Answer Relevancy | _fill in average from evaluation_results.json_ |

Faithfulness measures whether answers are grounded in retrieved context rather than hallucinated. Answer relevancy measures how well the answer actually addresses the question that was asked.

**A note on the relevancy score:** RAGAS's `answer_relevancy` metric normally generates 3 reverse-engineered questions per answer and averages across them, for stability. It does this by requesting `n=3` completions in a single API call. Groq's API doesn't support `n>1` at all — not a free-tier limit, just not supported — so this was set to `strictness=1` (1 generated question instead of 3). It's a real tradeoff: less statistical smoothing, slightly more sensitive to one unlucky generation. Documenting it here instead of pretending it's the default config.

## Known limitations

- **Answer relevancy runs at reduced strictness** because of the Groq `n>1` constraint above.
- **Multi-metric extraction can miss values that live in different sections of a long document.** Asking for revenue + net income + R&D in one go, for example, can come back with R&D as `None` even though it's in the document, because the top-k reranked chunks didn't happen to include the section it's reported in (income statement and R&D expense breakdown aren't always close together in a 130-page 10-K). Raising `top_k` for the extraction endpoint helps but doesn't fully solve it. A more complete fix would probably mean section-aware chunking or extracting once at upload time across the whole document instead of per-question — noted as a possible next step, not done here.

## Tech stack

- **LLM:** Mistral 7B via Ollama (fully local, no API cost)
- **Embeddings:** nomic-embed-text via Ollama
- **Vector store:** ChromaDB
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **Structured extraction:** Instructor + Groq (llama-3.3-70b-versatile)
- **Evaluation:** RAGAS + Groq (llama-3.3-70b-versatile)
- **Observability:** LangFuse
- **Backend:** FastAPI
- **Frontend:** Streamlit

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload and ingest a PDF |
| `/ask` | POST | Basic RAG query |
| `/ask/advanced` | POST | MultiQuery + HyDE + BM25 + rerank |
| `/ask/filtered` | POST | Filter by company and year |
| `/chat/memory` | POST | Conversational RAG with memory |
| `/evaluate` | POST | RAGAS evaluation scores |
| `/extract` | POST | Structured financial data extraction (Instructor) |

## How to run locally

Requirements: Python 3.10+, Ollama installed and running, Docker (for LangFuse).

Clone the repo:
```
git clone <your-repo-url>
cd advanced-rag-finance
```

Install dependencies:
```
pip install -r requirements.txt
```

Pull the required Ollama models:
```
ollama pull mistral
ollama pull nomic-embed-text
```

Add your Groq API key (free tier):
```
# Create .env file
GROQ_API_KEY=your_key_here
```

Start LangFuse (if you want tracing):
```
docker compose up -d
```

Start the backend:
```
uvicorn app.main:app --reload --port 8001
```

Start the frontend:
```
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, upload a PDF, and start asking questions.