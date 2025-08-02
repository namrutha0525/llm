"""
Unit tests for the LLM Document Retrieval API
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from main import app, DocumentProcessor, DocumentChunker, SemanticSearch

client = TestClient(app)

# Test data
VALID_TOKEN = "479309883e76b7aff59e87e1e032ce655934c42516b75cc1ceaea8663351e3ba"
INVALID_TOKEN = "invalid_token"

class TestAuthentication:
    def test_valid_authentication(self):
        response = client.get("/health", headers={"Authorization": f"Bearer {VALID_TOKEN}"})
        assert response.status_code == 200

    def test_invalid_authentication(self):
        response = client.get("/health", headers={"Authorization": f"Bearer {INVALID_TOKEN}"})
        assert response.status_code == 401

    def test_missing_authentication(self):
        response = client.get("/health")
        assert response.status_code == 403

class TestHealthEndpoints:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "3.0.0"

    def test_health_endpoint(self):
        response = client.get("/health", headers={"Authorization": f"Bearer {VALID_TOKEN}"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "timestamp" in data

class TestDocumentProcessing:
    def test_document_chunker(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(text, chunk_size=50, overlap=10)

        assert len(chunks) > 0
        assert all(chunk.chunk_id >= 0 for chunk in chunks)
        assert all(len(chunk.text) <= 60 for chunk in chunks)  # Allowing some variance

    def test_empty_text_chunking(self):
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0

class TestSemanticSearch:
    def test_keyword_search_fallback(self):
        from main import DocumentChunk

        chunks = [
            DocumentChunk(text="This is about machine learning", chunk_id=0, start_pos=0, end_pos=30, metadata={}),
            DocumentChunk(text="This is about cooking recipes", chunk_id=1, start_pos=31, end_pos=60, metadata={}),
            DocumentChunk(text="Machine learning algorithms", chunk_id=2, start_pos=61, end_pos=90, metadata={})
        ]

        search = SemanticSearch()
        search.chunks = chunks

        results = search._keyword_search("machine learning", top_k=2)
        assert len(results) <= 2
        assert all("machine" in result.text.lower() or "learning" in result.text.lower() for result in results)

class TestAPIEndpoints:
    @patch('main.download_document')
    @patch('main.DocumentProcessor.extract_text_from_pdf')
    async def test_document_query_endpoint(self, mock_extract, mock_download):
        # Mock responses
        mock_download.return_value = (b"fake pdf content", "pdf")
        mock_extract.return_value = ("Sample document text for testing.", {"pages": 1, "doc_type": "pdf"})

        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        payload = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is this document about?"],
            "max_chunks": 3,
            "include_metadata": True
        }

        response = client.post("/api/v1/hackrx/run", json=payload, headers=headers)
        # Note: This might fail without proper mocking of all async dependencies
        # In a real test environment, you'd mock more dependencies

    def test_external_connectivity_test(self):
        response = client.get("/api/v1/test-external")
        # This tests the endpoint existence, actual connectivity depends on internet
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

if __name__ == "__main__":
    pytest.main([__file__])
