"""Helpers for reading PDF and TXT files."""

from pathlib import Path

from pypdf import PdfReader


def _load_pdf(path: Path) -> str:
    """Extract text content from a PDF file."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _load_txt(path: Path) -> str:
    """Extract text content from a plain text file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document_text(path: str | Path) -> str:
    """Read a supported document and return its text."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix == ".txt":
        return _load_txt(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")
