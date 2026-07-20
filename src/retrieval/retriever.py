import itertools
from typing import List, Tuple, Optional, Dict

from langchain.schema import Document
from rank_bm25 import BM25Okapi

from ..vectorstore.chroma_store import chroma_store
from ..config import settings


class HybridRetriever:
    """Hybrid dense + BM25 retriever.

    * Dense retrieval uses Chroma similarity (lower distance = more similar).
    * Sparse retrieval uses BM25 on the raw chunk texts.
    * Results are merged via a simple weighted sum of normalized scores.
    """

    def __init__(self):
        # Load all documents once for BM25; keep in memory for fast sparse search.
        all_records = chroma_store.collection.get(include=["documents", "metadatas"])
        self.texts: List[str] = all_records.get("documents", [])
        self.metadatas: List[Dict] = all_records.get("metadatas", [])
        if self.texts:
            self.bm25 = BM25Okapi([doc.split() for doc in self.texts])
        else:
            self.bm25 = None

    def _dense_search(self, query: str, k: int, where: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        # Returns (Document, distance) where lower distance = more similar.
        if not self.texts:
            return []
        results = chroma_store.collection.query(
            query_texts=[query],
            n_results=k,
            where=where or {},
            include=["documents", "metadatas", "distances"],
        )
        docs_scores = []
        for doc_text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            docs_scores.append((Document(page_content=doc_text, metadata=meta), float(dist)))
        return docs_scores

    def _sparse_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        # Returns list of (index, score) sorted by BM25 score descending.
        if self.bm25 is None:
            return []
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        # Pair each score with its document index
        indexed_scores = list(enumerate(scores))
        # Sort by score descending (higher = more relevant)
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        use_hybrid: bool = True,
    ) -> List[Document]:
        """Retrieve top‑k documents using dense, sparse, or hybrid search.

        Args:
            query: User query string.
            top_k: Number of documents to return.
            metadata_filter: Optional Chroma ``where`` clause for filtering.
            use_hybrid: If ``False`` only the dense retriever is used.
        """
        dense_results = self._dense_search(query, top_k * 2, where=metadata_filter)
        if not use_hybrid:
            # Return the best dense results (sorted by distance ascending)
            dense_results.sort(key=lambda x: x[1])
            return [doc for doc, _ in dense_results[:top_k]
            ]

        # Sparse results (indices and BM25 scores)
        sparse_results = self._sparse_search(query, top_k * 2)

        # Normalize dense distances to similarity (higher = better)
        dense_scores = [1 / (dist + 1e-6) for _, dist in dense_results]
        max_dense = max(dense_scores) if dense_scores else 1.0
        dense_norm = [s / max_dense for s in dense_scores]

        # Normalize BM25 scores
        sparse_scores = [score for _, score in sparse_results]
        max_sparse = max(sparse_scores) if sparse_scores else 1.0
        sparse_norm = [s / max_sparse for s in sparse_scores]

        # Build a combined ranking using weighted sum
        weight_dense = settings.HYBRID_DENSE_WEIGHT
        weight_sparse = 1.0 - weight_dense

        # Map document index to combined score
        combined: Dict[int, float] = {}
        for (doc, _), ds in zip(dense_results, dense_norm):
            try:
                idx = self.texts.index(doc.page_content)
            except ValueError:
                # Should not happen, but guard against mismatches
                continue
            combined[idx] = combined.get(idx, 0) + weight_dense * ds
        for (idx, _), ss in zip(sparse_results, sparse_norm):
            combined[idx] = combined.get(idx, 0) + weight_sparse * ss

        # Sort indices by combined score descending and pick top_k
        sorted_idxs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        retrieved_docs = [Document(page_content=self.texts[i], metadata=self.metadatas[i]) for i, _ in sorted_idxs]
        return retrieved_docs

# Export a singleton for easy import
retriever = HybridRetriever()
