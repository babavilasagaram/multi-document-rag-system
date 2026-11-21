from app.database.connection import get_connection

chunk_ids = [74, 78, 79, 80]

with get_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, content
            FROM document_chunks
            WHERE id = ANY(%s)
            ORDER BY id;
            """,
            (chunk_ids,)
        )

        for chunk_id, content in cursor.fetchall():
            print("\n" + "=" * 80)
            print(f"CHUNK ID: {chunk_id}")
            print("=" * 80)
            print(content)