## ARCHITECTURE.md

### RAG Pipeline Overview
The system follows a classic Retrieval‑Augmented Generation flow:

```mermaid
flowchart TD
    User[User Query] -->|embed query| QueryEmbedding[Query Embedding]
    QueryEmbedding -->|retrieve top‑k| Retriever[Hybrid Retriever]
    Retriever -->|chunks + metadata| Retrieved[Retrieved Chunks]
    Retrieved -->|format context| Context[Prompt Context]
    Context -->|LLM prompt| LLM[LLM (Ollama / OpenAI)]
    LLM -->|generated answer| Answer[Answer Stream]
    Answer -->|map sources| Citations[Source Citations]
    Citations -->|display| UI[Streamlit UI]
```

1. **Embedding** – The query is embedded with the same model used for document chunks.
2. **Hybrid Retrieval** – Dense similarity (Chroma) + sparse BM25 scores are fused (RRR) to obtain the most relevant chunks.
3. **Prompt Construction** – Retrieved chunks are concatenated (respecting token limits) and fed into a prompt template that instructs the LLM to answer and cite sources.
4. **LLM Generation** – The LLM streams tokens back to the UI.
5. **Citation Mapping** – The pipeline returns the source chunk metadata alongside the answer, allowing the UI to render expandable citations.

### Optional Observability
When `LANGCHAIN_TRACING_V2=true` the entire flow is traced in LangSmith, capturing:
* Input query
* Retrieved chunk IDs and scores
* Prompt sent to the LLM
* Latency per step
* Token usage

---

The diagram and description above serve both as a design document and as personal study notes.
