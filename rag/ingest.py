"""Document ingestion logic for creating and storing embeddings."""

from pathlib import Path

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.document_loader import load_document_text
from utils.config import get_settings
from utils.endee_vector_store import get_vector_store


def _chunk_documents(raw_text: str, source_name: str) -> list[Document]:
    """Split raw text into smaller chunks that fit vector search and prompting."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)

    documents: list[Document] = []
    for index, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={"source": source_name, "chunk_id": index},
            )
        )
    return documents


def ingest_document(file_path: str | Path) -> dict:
    """Read a document, chunk it, embed it, and save it into the Endee-compatible storage layer."""
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Document not found: {path}")

    raw_text = load_document_text(path)
    if not raw_text.strip():
        raise ValueError("The document did not contain readable text.")

    documents = _chunk_documents(raw_text=raw_text, source_name=path.name)
    vector_store = get_vector_store()
    vector_store.add_documents(documents)

    return {
        "chunks_indexed": len(documents),
        "collection_name": get_settings().chroma_collection_name,
    }
