import os
from typing import List, Tuple, Dict

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from ..config import settings
from ..retrieval.retriever import retriever as hybrid_retriever
from ..vectorstore.chroma_store import chroma_store
from ..utils.citations import format_citations

# ---------------------------------------------------------------------------
# Helper: LLM factory based on settings
# ---------------------------------------------------------------------------
def _get_llm():
    provider = settings.LLM_PROVIDER.lower()
    model_name = settings.LLM_MODEL
    if provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(model=model_name, base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
    elif provider == "openai":
        from langchain_community.llms import OpenAI
        return OpenAI(model_name=model_name, openai_api_key=settings.OPENAI_API_KEY, temperature=0.0)
    elif provider == "anthropic":
        from langchain_community.llms import Anthropic
        return Anthropic(model=model_name, anthropic_api_key=settings.ANTHROPIC_API_KEY, temperature=0.0)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# ---------------------------------------------------------------------------
# Prompt template – instruct the model to answer and cite sources.
# ---------------------------------------------------------------------------
_PROMPT = """
You are an AI assistant answering a user query based on the provided context.

Context (the most relevant document chunks, each preceded by a source marker):
{context}

---
Answer the question below **using only information from the context**. Cite the source(s) by including the marker you see in the context (e.g., [source 1]).

Question: {question}
"""
prompt_template = PromptTemplate.from_template(_PROMPT)

# ---------------------------------------------------------------------------
# Context formatter – takes retrieved documents and builds a numbered source list.
# ---------------------------------------------------------------------------
def _format_context(docs: List[Document]) -> str:
    """Create a single string with numbered source blocks.

    Each block is prefixed with ``[source i]`` where *i* is a 1‑based index.
    The block includes the raw chunk text.
    """
    parts = []
    for idx, doc in enumerate(docs, start=1):
        source_meta = doc.metadata
        source_label = f"[source {idx}]"
        # Include filename and page for readability (optional)
        extra = []
        if source_meta.get("source"):
            extra.append(source_meta["source"])
        if "page" in source_meta:
            extra.append(f"page {source_meta['page']}")
        header = f"{source_label} ({', '.join(extra)})" if extra else source_label
        parts.append(f"{header}\n{doc.page_content.strip()}\n")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Build the LCEL chain
# ---------------------------------------------------------------------------
def build_chain():
    # 1️⃣ Retriever (dense + optional hybrid) – returns List[Document]
    retriever_node = RunnableLambda(
        lambda query: hybrid_retriever.retrieve(
            query,
            top_k=settings.TOP_K,
            metadata_filter=None,  # UI will pass filter via RunnablePassthrough later
            use_hybrid=True,
        )
    )

    # 2️⃣ Pass‑through the query alongside the retrieved docs so we can keep both.
    #    We'll later format the context using the docs.
    prep = RunnableParallel({
        "docs": retriever_node,
        "question": RunnablePassthrough(),
    })

    # 3️⃣ Build the final prompt input.
    def _assemble(inputs: Dict) -> Dict:
        ctx = _format_context(inputs["docs"])
        return {"context": ctx, "question": inputs["question"]}

    chain = (
        prep
        | RunnableLambda(_assemble)
        | prompt_template
        | _get_llm()
        | StrOutputParser()
    )
    return chain

# Export a singleton for easy use in the UI
rag_chain = build_chain()

# ---------------------------------------------------------------------------
# Convenience wrappers used by the Streamlit UI
# ---------------------------------------------------------------------------
def invoke(query: str) -> str:
    """Run the chain synchronously and return the full answer string."""
    return rag_chain.invoke(query)

def stream(query: str):
    """Yield tokens (or chunks) from the chain's streaming interface.

    The caller can iterate over the generator and write each chunk to the UI.
    """
    return rag_chain.stream(query)
