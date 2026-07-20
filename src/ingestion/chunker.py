import itertools
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.schema import Document

from ..config import settings


def split_fixed_size(docs: List[Document]) -> List[Document]:
    """Split documents using a fixed-size character splitter with overlap.

    The splitter respects the ``CHUNK_SIZE`` and ``CHUNK_OVERLAP`` settings.
    Each output Document inherits the original metadata and adds a ``chunk_index``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
    )
    split_docs: List[Document] = []
    for doc in docs:
        chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(chunks):
            # Preserve source metadata and add chunk identifier
            new_meta = dict(chunk.metadata)
            new_meta.update({"chunk_index": idx})
            split_docs.append(Document(page_content=chunk.page_content, metadata=new_meta))
    return split_docs


def split_by_headers(docs: List[Document]) -> List[Document]:
    """Semantic/structure‑aware splitter that tries to split on common headings.

    It looks for lines that resemble Markdown or simple section headers (e.g., "1. Introduction").
    If no header is found in a document, it falls back to the fixed‑size splitter.
    """
    def has_header(text: str) -> bool:
        # Very lightweight heuristic: line starts with a number or all‑caps word followed by a colon
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[:2].isdigit() and (stripped[2] == "." or stripped[2] == " "):
                return True
            if stripped.isupper() and len(stripped.split()) <= 5:
                return True
            if stripped.endswith(":"):
                return True
        return False

    split_docs: List[Document] = []
    for doc in docs:
        if has_header(doc.page_content):
            # Use newline as separator, but keep large chunks by merging until size limit
            lines = doc.page_content.splitlines()
            current_chunk = []
            current_len = 0
            chunk_idx = 0
            for line in itertools.chain(lines, ["---END---"]):
                # ``---END---`` forces the final flush
                line_len = len(line) + 1  # include newline
                if current_len + line_len > settings.CHUNK_SIZE and current_chunk:
                    # Flush current chunk
                    new_meta = dict(doc.metadata)
                    new_meta.update({"chunk_index": chunk_idx})
                    split_docs.append(Document(page_content="\n".join(current_chunk), metadata=new_meta))
                    chunk_idx += 1
                    current_chunk = []
                    current_len = 0
                if line != "---END---":
                    current_chunk.append(line)
                    current_len += line_len
        else:
            # Fallback to fixed size splitter for this document
            split_docs.extend(split_fixed_size([doc]))
    return split_docs
