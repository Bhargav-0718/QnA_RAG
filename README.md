# RAG Agent (contract Q&A)

This repository contains a small Retrieval-Augmented Generation (RAG) agent that answers questions about a commercial contract PDF.

Features
- Uses LangChain utilities for text splitting and vectorstore wiring.
- Uses OpenAI embeddings (text-embedding-3-small) and ChromaDB persisted to `chroma_db/`.
- Four-node workflow (plan → retrieve → answer → reflect) implemented in `src/agent.py`.
- Optional Streamlit UI in `src/ui.py`.

Requirements
- Python 3.8+
- Set your OpenAI API key in the environment: `OPENAI_API_KEY`.
- Install dependencies (see `requirements.txt`).

Quick start
1. Install dependencies (preferably inside a virtualenv):

```powershell
python -m pip install -r requirements.txt
```

2. Place your contract PDF at `data/contract.pdf` (or pass `--pdf PATH` to the script).

3. Run the agent from the project root:

```powershell
python src/agent.py --query "What is the termination clause?"
```

4. (Optional) Start the Streamlit UI:

```powershell
streamlit run src/ui.py
```

Notes
- The first run will extract text and create embeddings; this can take a few minutes depending on the document size and network latency to the OpenAI embeddings service.
- LangSmith tracing is enabled by setting the environment variables in `src/agent.py` (you can also set `LANGCHAIN_TRACING_V2` and `LANGCHAIN_PROJECT` externally).
- If `langgraph` isn't installed, the script falls back to a direct sequential workflow while still satisfying the plan/retrieve/answer/reflect contract.

Environment variables
- OPENAI_API_KEY — required for embeddings and LLM calls.
- LANGCHAIN_TRACING_V2 — set to `true` to enable tracing.
- LANGCHAIN_PROJECT — project name for tracing (default `rag-agent`).

Files of interest
- `src/rag_utils.py` — PDF extraction, vectorstore creation and retrieval.
- `src/agent.py` — main RAG workflow (plan/retrieve/answer/reflect).
- `src/ui.py` — minimal Streamlit UI.

License
MIT
