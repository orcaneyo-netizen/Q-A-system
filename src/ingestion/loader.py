import os
from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain.schema import Document

from ..config import settings


def load_pdfs() -> List[Document]:
    """Load all PDFs from the configured raw PDF directory.

    Returns:
        List[Document]: A list of LangChain Document objects, each representing a
        single page of a PDF. Metadata includes:
            - "source": filename
            - "page": page number (1‑indexed)
    """
    pdf_dir = Path(settings.RAW_PDF_PATH)
    documents: List[Document] = []

    if not pdf_dir.exists():
        raise FileNotFoundError(f"Raw PDF directory not found: {pdf_dir}")

    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as e:
            # Skip files that cannot be read but continue processing others
            print(f"[loader] Failed to read {pdf_path.name}: {e}")
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"[loader] Error extracting text from {pdf_path.name} page {page_number}: {e}")
                text = ""
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_number,
                },
            )
            documents.append(doc)
    return documents
