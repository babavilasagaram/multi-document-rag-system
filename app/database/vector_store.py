
import json

from app.database.connection import get_connection


def store_embeddings(embedded_chunks, source, content_hash):
    """
    Store document chunks with idempotent document ingestion.

    Returns:
        dict: Ingestion status and chunk count.
    """

    if not embedded_chunks:
        raise ValueError("No chunks provided for storage.")

    if len(content_hash) != 64:
        raise ValueError("Invalid SHA-256 hash.")

    if not source:
        raise ValueError("Document source cannot be empty.")

    with get_connection() as conn:
        with conn.cursor() as cursor:

            # 1. Register the document if it does not exist.
            cursor.execute(
                """
                INSERT INTO documents (source, content_hash)
                VALUES (%s, %s)
                ON CONFLICT (source) DO NOTHING;
                """,
                (source, content_hash)
            )

            # 2. Lock the document row to prevent concurrent replacement.
            cursor.execute(
                """
                SELECT document_id, content_hash
                FROM documents
                WHERE source = %s
                FOR UPDATE;
                """,
                (source,)
            )

            document_id, existing_hash = cursor.fetchone()

            # 3. Skip unchanged documents.
            if existing_hash.strip() == content_hash:
                # A newly inserted document also has this hash.
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM document_chunks
                    WHERE document_id = %s;
                    """,
                    (document_id,)
                )

                existing_count = cursor.fetchone()[0]

                if existing_count > 0:
                    return {
                        "status": "SKIPPED",
                        "source": source,
                        "chunks": existing_count
                    }

            else:
                # 4. Remove old chunks for a modified document.
                cursor.execute(
                    """
                    DELETE FROM document_chunks
                    WHERE document_id = %s;
                    """,
                    (document_id,)
                )

                cursor.execute(
                    """
                    UPDATE documents
                    SET content_hash = %s,
                        updated_at = NOW()
                    WHERE document_id = %s;
                    """,
                    (content_hash, document_id)
                )

            # 5. Insert new chunks.
            for chunk_index, chunk in enumerate(embedded_chunks):

                vector = (
                    "["
                    + ",".join(map(str, chunk["embedding"]))
                    + "]"
                )

                metadata = {
                    **chunk["metadata"],
                    "source": source
                }

                cursor.execute(
                    """
                    INSERT INTO document_chunks
                    (
                        document_id,
                        chunk_index,
                        content,
                        metadata,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s::jsonb, %s::vector);
                    """,
                    (
                        document_id,
                        chunk_index,
                        chunk["text"],
                        json.dumps(metadata),
                        vector
                    )
                )

    print(
        f"Stored {len(embedded_chunks)} chunks "
        f"for document: {source}"
    )

    return {
        "status": "STORED",
        "source": source,
        "chunks": len(embedded_chunks)
    }
