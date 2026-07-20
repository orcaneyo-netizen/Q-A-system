# Local RAG Document Q&A System

A production‑quality Retrieval‑Augmented Generation (RAG) reference project that ingests PDFs, stores embeddings in a local ChromaDB, retrieves relevant chunks (dense and optional hybrid search), and answers queries via a configurable LLM with streaming responses and source citations.

## Features
- PDF ingestion with page‑level metadata
- Configurable chunking (fixed‑size & header‑aware)
- Swappable embeddings: local `sentence‑transformers` or OpenAI
- Persistent ChromaDB vector store
- Hybrid dense + BM25 retrieval (RRR fusion)
- LCEL‑based LangChain pipeline (retriever → formatter → prompt → LLM → parser)
- Token‑level streaming UI built with Streamlit
- Optional LangSmith tracing for observability
- Expandable source citations for every answer

## Quick Start
```bash
# Clone / copy this repo
cd rag-qa-system

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy example env and edit values
cp .env.example .env
# edit .env as needed (LLM keys, embedding model, etc.)

# Run the app
streamlit run app.py
```

Place PDFs under `data/raw_pdfs/` or upload them via the UI.

## Project Structure
```
rag-qa-system/
├─ app.py                     # Streamlit entrypoint
├─ requirements.txt
├─ .env.example
├─ README.md
├─ ARCHITECTURE.md
├─ src/
│  ├─ config.py
│  ├─ ingestion/
│  │   ├─ loader.py
│  │   └─ chunker.py
│  ├─ vectorstore/
│  │   └─ chroma_store.py
│  ├─ retrieval/
│  │   └─ retriever.py
│  ├─ chains/
│  │   └─ rag_chain.py
│  └─ utils/
│      └─ citations.py
├─ data/
│  ├─ raw_pdfs/               # PDF files to ingest
│  └─ chroma_db/              # Persistent Chroma collection
```

For detailed architecture, see `ARCHITECTURE.md`.
