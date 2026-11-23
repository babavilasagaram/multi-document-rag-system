
import hashlib
import logging
import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.database.connection import get_connection
from app.ingestion.pipeline import (
    ingest_pdf,
    calculate_file_hash,
    EXTRACTION_VERSION,
)
from app.api.schemas.documents import (
    DocumentUploadResponse,
    DocumentItem,
    DocumentListResponse,
)


logger = logging.getLogger(__name__)

MAX_FILE_SIZE = int(
    os.getenv("MAX_UPLOAD_SIZE_MB", "20")
) * 1024 * 1024

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


# --------------------------------------------------
# POST /documents/upload
# Upload and ingest a PDF
# --------------------------------------------------

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
def upload_document(file: UploadFile = File(...)):

    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    if file.file.read(5) != b"%PDF-":
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file.",
        )

    file.file.seek(0)

    file_path = None

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        stored_filename = f"{uuid4().hex}.pdf"
        file_path = UPLOAD_DIR / stored_filename

        total_size = 0

        # Save the uploaded PDF in chunks.
        with file_path.open("wb") as destination:

            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the upload size limit.",
                    )

                destination.write(chunk)

        # Calculate the same versioned hash used by ingestion.
        pdf_hash = calculate_file_hash(file_path)

        content_hash = hashlib.sha256(
            f"{pdf_hash}:{EXTRACTION_VERSION}".encode()
        ).hexdigest()

        # Check whether this document is already indexed.
        with get_connection() as conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT
                        d.source,
                        COUNT(c.id) AS chunks
                    FROM documents d
                    LEFT JOIN document_chunks c
                        ON d.document_id = c.document_id
                    WHERE d.content_hash = %s
                    GROUP BY d.document_id, d.source;
                    """,
                    (content_hash,),
                )

                duplicate = cursor.fetchone()

        if duplicate:
            return DocumentUploadResponse(
                filename=Path(file.filename).name,
                status="SKIPPED",
                chunks=duplicate[1],
                message="This document is already indexed.",
            )

        # Ingest new documents.
        result = ingest_pdf(file_path)

        return DocumentUploadResponse(
            filename=Path(file.filename).name,
            status=result["status"],
            chunks=result["chunks"],
            message="Document processed successfully.",
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Document upload or ingestion failed.")

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        )

    finally:
        # Always remove the temporary PDF after processing.
        if file_path is not None:
            file_path.unlink(missing_ok=True)


# --------------------------------------------------
# GET /documents
# Retrieve all indexed documents
# --------------------------------------------------

@router.get(
    "",
    response_model=DocumentListResponse,
)
def list_documents():

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT
                        d.source,
                        COUNT(c.id) AS chunks
                    FROM documents d
                    LEFT JOIN document_chunks c
                        ON d.document_id = c.document_id
                    GROUP BY
                        d.document_id,
                        d.source,
                        d.updated_at
                    ORDER BY d.updated_at DESC;
                    """
                )

                rows = cursor.fetchall()

        documents = [
            DocumentItem(
                filename=Path(source).name,
                chunks=chunks,
            )
            for source, chunks in rows
        ]

        return DocumentListResponse(
            documents=documents,
            total=len(documents),
        )

    except Exception:
        logger.exception("Failed to retrieve documents.")

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve documents.",
        )
