
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import documents as documents_route
from app.api.routes import query as query_route


client = TestClient(app)


def mock_connection(fetchone=None, fetchall=None):
    """Create a mocked database connection and cursor."""

    connection = MagicMock()
    cursor_context = MagicMock()
    cursor = MagicMock()

    connection.__enter__.return_value = connection
    connection.cursor.return_value = cursor_context
    cursor_context.__enter__.return_value = cursor

    cursor.fetchone.return_value = fetchone
    cursor.fetchall.return_value = fetchall or []

    return connection


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# --------------------------------------------------
# Document listing
# --------------------------------------------------

def test_list_documents():

    rows = [
        ("data/example.pdf", 10),
        ("data/another.pdf", 20)
    ]

    connection = mock_connection(fetchall=rows)

    with patch.object(
        documents_route,
        "get_connection",
        return_value=connection
    ):
        response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert len(data["documents"]) == 2
    assert data["documents"][0]["filename"] == "example.pdf"
    assert data["documents"][0]["chunks"] == 10


# --------------------------------------------------
# PDF upload validation
# --------------------------------------------------

def test_upload_rejects_non_pdf():

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                b"Some text",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_upload_rejects_invalid_pdf():

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "invalid.pdf",
                b"Not a PDF file",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 400


# --------------------------------------------------
# Duplicate upload
# --------------------------------------------------

def test_duplicate_upload_is_skipped(tmp_path):

    monkeypatch_upload_dir = tmp_path

    connection = mock_connection(
        fetchone=("data/existing.pdf", 83)
    )

    with (
        patch.object(
            documents_route,
            "UPLOAD_DIR",
            monkeypatch_upload_dir
        ),
        patch.object(
            documents_route,
            "calculate_file_hash",
            return_value="a" * 64
        ),
        patch.object(
            documents_route,
            "get_connection",
            return_value=connection
        ),
        patch.object(
            documents_route,
            "ingest_pdf"
        ) as mock_ingest
    ):

        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "duplicate.pdf",
                    b"%PDF-test content",
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "SKIPPED"
    assert data["chunks"] == 83

    # Duplicate must not be ingested again.
    mock_ingest.assert_not_called()

    # Temporary uploaded copy must be removed.
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------
# New PDF upload
# --------------------------------------------------

def test_new_upload_is_ingested(tmp_path):

    connection = mock_connection(fetchone=None)

    with (
        patch.object(
            documents_route,
            "UPLOAD_DIR",
            tmp_path
        ),
        patch.object(
            documents_route,
            "calculate_file_hash",
            return_value="b" * 64
        ),
        patch.object(
            documents_route,
            "get_connection",
            return_value=connection
        ),
        patch.object(
            documents_route,
            "ingest_pdf",
            return_value={
                "status": "STORED",
                "chunks": 25
            }
        ) as mock_ingest
    ):

        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "new_document.pdf",
                    b"%PDF-test content",
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "STORED"
    assert data["chunks"] == 25

    mock_ingest.assert_called_once()

    # Successfully uploaded file remains stored.
    assert len(list(tmp_path.iterdir())) == 0


# --------------------------------------------------
# Query endpoint
# --------------------------------------------------

def test_query_answerable_question():

    retrieved_chunks = [
        {
            "id": 1,
            "text": "The Transformer uses eight attention heads.",
            "metadata": {
                "source": "data/transformer.pdf",
                "page": 5
            },
            "similarity": 0.75
        }
    ]

    generated_answer = (
        "The Transformer uses eight attention heads. "
        "[Source: data/transformer.pdf, Page: 5]"
    )

    with (
        patch.object(
            query_route,
            "retrieve_documents",
            return_value=retrieved_chunks
        ),
        patch.object(
            query_route,
            "generate_answer",
            return_value=generated_answer
        )
    ):

        response = client.post(
            "/query",
            json={
                "question": "How many attention heads does the Transformer use?"
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == generated_answer

    assert data["sources"] == [
        {
            "filename": "transformer.pdf",
            "page": 5
        }
    ]


def test_query_unanswerable_question():

    abstention = (
        "I cannot find the answer in the provided documents."
    )

    with (
        patch.object(
            query_route,
            "retrieve_documents",
            return_value=[]
        ),
        patch.object(
            query_route,
            "generate_answer",
            return_value=abstention
        )
    ):

        response = client.post(
            "/query",
            json={
                "question": "Who won the FIFA World Cup in 1998?"
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == abstention
    assert data["sources"] == []


def test_query_rejects_empty_question():

    response = client.post(
        "/query",
        json={
            "question": ""
        }
    )

    assert response.status_code == 422
