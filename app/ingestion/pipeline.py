
import hashlib
from pathlib import Path

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_documents
from app.embeddings.embedder import embed_documents
from app.database.vector_store import store_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTION_VERSION = "v2"

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of the original PDF."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def ingest_pdf(file_path):
    """Load, chunk, embed and store a PDF idempotently."""

    pdf_path = Path(file_path).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Create a stable document source.
    try:
        source = pdf_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source = pdf_path.as_posix()

    # Identify the exact PDF version.
    pdf_hash = calculate_file_hash(pdf_path)

    content_hash = hashlib.sha256(
        f"{pdf_hash}:{EXTRACTION_VERSION}".encode()
    ).hexdigest()

    # Load PDF.
    documents = load_pdf(str(pdf_path))

    # Preserve the normalized source path.
    for document in documents:
        document["metadata"]["source"] = source

    # Chunk documents.
    chunks = chunk_documents(documents)

    # Generate embeddings.
    embedded_chunks = embed_documents(chunks)

    # Store or replace document.
    result = store_embeddings(
        embedded_chunks=embedded_chunks,
        source=source,
        content_hash=content_hash
    )

    return result

def ingest_directory(directory_path):
    directory = Path(directory_path).resolve()

    if not directory.is_dir():
        raise NotADirectoryError(directory)

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF files found.")

    results = []

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        try:
            result = ingest_pdf(pdf_file)
            results.append(result)

        except Exception as e:
            print(f"Failed: {pdf_file.name}: {e}")

            results.append({
                "status": "FAILED",
                "source": pdf_file.name,
                "error": str(e)
            })

    return results