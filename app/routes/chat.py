"""Chat routes for question answering."""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.chat_service import answer_question, get_supported_models

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    """Input payload for a chat request."""

    question: str = Field(..., min_length=1, description="User question")
    session_id: str = Field(default="default", min_length=1, description="Chat session identifier")
    model_name: Literal["openai", "extractive"] = Field(
        default="openai",
        description="Generator choice",
    )
    top_k: int = Field(default=4, ge=1, le=8, description="Number of chunks to retrieve")


@router.get("/models")
def list_models() -> dict:
    """Return generator choices that the frontend can display."""
    return {"models": get_supported_models()}


@router.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Answer a user question using retrieval-augmented generation."""
    try:
        return answer_question(
            question=request.question,
            session_id=request.session_id,
            model_name=request.model_name,
            top_k=request.top_k,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat request failed. Check the backend logs for details.",
        )


