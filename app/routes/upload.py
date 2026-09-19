"""Upload routes for document ingestion."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.services.document_service import process_uploaded_file

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Accept a document and index it before returning success."""
    try:
        return await process_uploaded_file(file)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception:
        logger.exception("Upload processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process upload. Check the backend logs for details.",
        )
