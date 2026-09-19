"""Document service for file validation and ingestion."""

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from rag.ingest import ingest_document
from utils.config import get_settings

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024


def _validate_file_type(filename: str) -> str:
    """Validate the uploaded file extension and return it."""
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise ValueError("Only PDF and TXT files are supported.")
    return extension


async def process_uploaded_file(file: UploadFile) -> dict:
    """Save an uploaded file, index it, and return only after indexing succeeds."""
    if not file.filename:
        raise ValueError("Uploaded file must have a filename.")

    extension = _validate_file_type(file.filename)
    settings = get_settings()

    safe_name = Path(file.filename).name
    content = await file.read()
    if not content:
        raise ValueError("Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError("Uploaded file is too large. Maximum size is 10 MB.")

    temp_name = f"{uuid4().hex}_{safe_name}"
    temp_path = Path(settings.upload_dir) / temp_name
    temp_path.write_bytes(content)

    try:
        result = ingest_document(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "message": "Document indexed successfully.",
        "filename": safe_name,
        "file_type": extension,
        **result,
    }
