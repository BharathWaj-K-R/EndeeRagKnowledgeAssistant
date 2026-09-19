import unittest
from unittest.mock import patch

from langchain.schema import Document

from app.services.document_service import MAX_UPLOAD_SIZE_BYTES, _validate_file_type
from rag.generator import _extractive_answer, generate_answer
from rag.ingest import _chunk_documents


class DocumentServiceTests(unittest.TestCase):
    def test_supported_extensions(self):
        self.assertEqual(_validate_file_type("notes.pdf"), ".pdf")
        self.assertEqual(_validate_file_type("notes.TXT"), ".txt")

    def test_rejects_unsupported_extension(self):
        with self.assertRaises(ValueError):
            _validate_file_type("notes.docx")

    def test_upload_limit_is_ten_megabytes(self):
        self.assertEqual(MAX_UPLOAD_SIZE_BYTES, 10 * 1024 * 1024)


class IngestionTests(unittest.TestCase):
    def test_chunking_returns_documents(self):
        documents = _chunk_documents("one two three four", "notes.txt")
        self.assertTrue(documents)
        self.assertEqual(documents[0].metadata["source"], "notes.txt")
        self.assertEqual(documents[0].metadata["chunk_id"], 0)


class GeneratorTests(unittest.TestCase):
    def test_extractive_answer_without_documents(self):
        answer = _extractive_answer("What?", [])
        self.assertIn("could not find relevant information", answer.lower())

    def test_extractive_answer_uses_retrieved_text(self):
        documents = _chunk_documents("Python is a programming language.", "notes.txt")
        answer = _extractive_answer("What is Python?", documents)
        self.assertIn("Python is a programming language.", answer)

    def test_missing_openai_key_uses_fallback(self):
        document = Document(
            page_content="Python is a programming language.",
            metadata={"source": "notes.txt", "chunk_id": 0},
        )
        with patch("rag.generator.get_settings") as get_settings:
            get_settings.return_value.openai_api_key = None
            answer, sources, mode = generate_answer(
                question="What is Python?",
                documents=[document],
                chat_history=[],
                model_name="openai",
            )

        self.assertIn("Python is a programming language.", answer)
        self.assertEqual(sources[0]["source"], "notes.txt")
        self.assertEqual(mode, "extractive_fallback")


if __name__ == "__main__":
    unittest.main()
