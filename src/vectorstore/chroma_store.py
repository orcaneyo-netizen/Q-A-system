import os
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from langchain.schema import Document

from ..config import settings

# Initialize embedding function – we will use the same sentence‑transformers model as the document embedder.
# This function can be swapped later if needed (e.g., OpenAI embeddings).

def _get_embedding_fn():
    # Use sentence‑transformers via embedding_functions.SentenceTransformerEmbeddingFunction
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.EMBEDDING_MODEL)


class ChromaStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="rag_docs",
            embedding_function=_get_embedding_fn(),
        )

    def _doc_to_record(self, doc: Document, idx: int) -> Tuple[str, str, dict]:
        """Convert a LangChain Document to Chroma record elements.
        Returns (id, text, metadata).
        """
        doc_id = f"doc_{idx}"
        metadata = doc.metadata.copy()
        # Ensure we store source, page, and chunk_index for citations
        metadata.setdefault("source", "unknown")
        metadata.setdefault("page", -1)
        metadata.setdefault("chunk_index", -1)
        return doc_id, doc.page_content, metadata

    def add_documents(self, docs: List[Document]):
        """Add a list of Documents to the collection.
        This will replace any existing records with the same IDs.
        """
        ids, texts, metadatas = [], [], []
        for idx, doc in enumerate(docs):
            doc_id, text, meta = self._doc_to_record(doc, idx)
            ids.append(doc_id)
            texts.append(text)
            metadatas.append(meta)
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 5, where: dict | None = None) -> List[Document]:
        """Perform a dense similarity search.
        Returns a list of Documents (without scores).
        """
        results = self.collection.query(
            query_texts=[query], n_results=k, where=where or {}, include=['documents', 'metadatas']
        )
        docs: List[Document] = []
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    def similarity_search_with_score(self, query: str, k: int = 5, where: dict | None = None) -> List[Tuple[Document, float]]:
        """Dense search returning (Document, score) pairs.
        ``score`` is the distance returned by Chroma (lower = more similar).
        """
        results = self.collection.query(
            query_texts=[query], n_results=k, where=where or {}, include=['documents', 'metadatas', 'distances']
        )
        docs_scores: List[Tuple[Document, float]] = []
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            docs_scores.append((Document(page_content=text, metadata=meta), float(dist)))
        return docs_scores

    def rebuild_index(self, docs: List[Document]):
        """Delete the existing collection and re‑add all documents.
        Useful when chunking parameters change.
        """
        self.client.delete_collection(name="rag_docs")
        # Re‑create collection
        self.collection = self.client.get_or_create_collection(
            name="rag_docs",
            embedding_function=_get_embedding_fn(),
        )
        self.add_documents(docs)

# Export a singleton for convenient import
chroma_store = ChromaStore()
