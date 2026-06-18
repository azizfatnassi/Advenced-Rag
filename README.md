Advanced RAG Finance Assistant

## What it does
A conversational AI assistant for financial documents. Upload any financial PDF
(annual reports, earnings, filings) and ask questions in plain English.
The system remembers your conversation so you can ask follow-up questions
naturally without repeating context.

Built for analysts, investors, or anyone who needs to extract insights from
financial documents without reading hundreds of pages manually.

## Architecture

User → Streamlit UI → FastAPI Backend → Retrieval Pipeline → Mistral 7B → Answer

↓

1. MultiQuery (3 variations)

2. HyDE (hypothetical answer)

3. BM25 (keyword search)

4. Merge all results

5. CrossEncoder Reranking

6. Memory injection

7. Mistral generates answer


## Techniques Used and WHY

**MultiQuery Retrieval** — One question = one angle = missed chunks.
Mistral generates 3 variations of your question to cover more semantic ground.

**HyDE (Hypothetical Document Embeddings)** — Questions and answers live in
different vector spaces. We ask Mistral to write a hypothetical answer first,
then search with that. Answer-to-answer similarity is more accurate than
question-to-answer.

**BM25 Hybrid Search** — Semantic search understands meaning but misses exact
terms. BM25 catches exact matches like ticker names, figures, and quarter labels.
Finance documents need both.

**CrossEncoder Reranking** — Similarity search retrieves 20 candidates fast.
The cross-encoder then reads question and chunk together to score true relevance.
Slower but much more accurate on the final top 3.

**Conversation-Aware Retrieval** — Follow-up questions like "how does that
compare?" are meaningless without context. We combine chat history with the
current question before searching ChromaDB.

**Session Memory** — Each conversation has a session ID. History is stored
and injected into every prompt so Mistral maintains context across turns.

## Benchmark Results
Evaluated on 10 financial questions from Tesla 2023 Annual Report:

| Metric | Score |
|--------|-------|
| Faithfulness | 0.802 / 1.0 |

Faithfulness measures whether answers are grounded in retrieved context
rather than hallucinated.

## Tech Stack
- **LLM**: Mistral 7B via Ollama (fully local, no API cost)
- **Embeddings**: nomic-embed-text via Ollama
- **Vector Store**: ChromaDB
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Evaluation**: RAGAS + Groq (llama-3.3-70b)
- **Backend**: FastAPI
- **Frontend**: Streamlit

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /upload | POST | Upload and ingest a PDF |
| /ask | POST | Basic RAG query |
| /ask/advanced | POST | MultiQuery + HyDE + BM25 + Rerank |
| /ask/filtered | POST | Filter by company and year |
| /chat/memory | POST | Conversational RAG with memory |
| /evaluate | POST | RAGAS evaluation scores |

## How to Run Locally

**Requirements**: Python 3.10+, Ollama installed and running

1. Clone the repo
```bash
git clone <your-repo-url>
cd advanced-rag-finance
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Pull required Ollama models
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

4. Add your Groq API key (free tier)
```bash
# Create .env file
GROQ_API_KEY=your_key_here
```

5. Start the backend
```bash
uvicorn app.main:app --reload --port 8001
```

6. Start the frontend
```bash
streamlit run streamlit_app.py
```

7. Open http://localhost:8501, upload a PDF and start asking questions.