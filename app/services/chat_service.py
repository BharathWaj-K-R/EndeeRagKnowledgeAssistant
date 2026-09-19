"""Chat service for RAG orchestration."""

from rag.generator import generate_answer
from rag.retriever import retrieve_relevant_chunks
from utils.memory import memory_store


def get_supported_models() -> list[dict]:
    """Return the generator options available to the UI."""
    return [
        {
            "label": "OpenAI Chat Model",
            "value": "openai",
            "description": "Best answer quality when OPENAI_API_KEY is configured.",
        },
        {
            "label": "Local Extractive Fallback",
            "value": "extractive",
            "description": "Low-resource mode that answers from retrieved chunks only.",
        },
    ]


def answer_question(
    question: str,
    session_id: str,
    model_name: str,
    top_k: int,
) -> dict:
    """Run the full RAG flow and store chat history for the given session."""
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    history = memory_store.get_history(session_id=session_id)
    documents = retrieve_relevant_chunks(query=cleaned_question, top_k=top_k)
    answer, sources, answer_mode = generate_answer(
        question=cleaned_question,
        documents=documents,
        chat_history=history,
        model_name=model_name,
    )
    memory_store.add_message(session_id=session_id, role="user", content=cleaned_question)
    memory_store.add_message(session_id=session_id, role="assistant", content=answer)

    return {
        "answer": answer,
        "answer_mode": answer_mode,
        "sources": sources,
        "history": memory_store.get_history(session_id=session_id),
    }
