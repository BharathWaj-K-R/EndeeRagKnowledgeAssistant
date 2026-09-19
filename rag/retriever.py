"""Retriever logic for semantic search."""

from langchain.schema import Document

from utils.config import get_settings
from utils.endee_vector_store import get_vector_store


def retrieve_relevant_chunks(query: str, top_k: int = 4) -> list[Document]:
    """Retrieve the most semantically relevant chunks for a question."""
    vector_store = get_vector_store()
    settings = get_settings()
    return vector_store.similarity_search(
        query=query,
        top_k=top_k,
        score_threshold=settings.similarity_threshold,
    )
