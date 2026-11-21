
from app.ingestion.chunker import chunk_documents


def test_chunk_documents_returns_chunks():
    documents = [
        {
            "text": "Transformers use attention mechanisms. " * 100,
            "metadata": {
                "source": "test.pdf",
                "page": 1
            }
        }
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) > 1
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)


def test_chunk_documents_preserves_metadata():
    metadata = {
        "source": "test.pdf",
        "page": 3
    }

    documents = [
        {
            "text": "Attention is important.",
            "metadata": metadata
        }
    ]

    chunks = chunk_documents(documents)

    assert chunks[0]["metadata"] == metadata


def test_chunk_documents_respects_chunk_size():
    documents = [
        {
            "text": "A" * 3000,
            "metadata": {
                "source": "test.pdf",
                "page": 1
            }
        }
    ]

    chunks = chunk_documents(documents)

    assert all(len(chunk["text"]) <= 800 for chunk in chunks)
