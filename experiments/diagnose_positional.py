
from app.database.connection import get_connection
from app.embeddings.embedder import embed_query

query = {
    "query": "Why does Transformer need positional encoding?",
    "relevant_ids": [79]
}
query_embedding = embed_query(query)
vector = "[" + ",".join(map(str, query_embedding)) + "]"

with get_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                id,
                metadata,
                content,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector, id;
            """,
            (vector, vector)
        )

        results = cursor.fetchall()

print("\nTOP 10 RETRIEVED CHUNKS")
print("=" * 80)

for rank, row in enumerate(results[:10], start=1):
    chunk_id, metadata, content, similarity = row

    print(f"\nRank: {rank}")
    print(f"Chunk ID: {chunk_id}")
    print(f"Page: {metadata.get('page')}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Content:\n{content[:900]}")


print("\nKNOWN RELEVANT CHUNKS")
print("=" * 80)

for rank, row in enumerate(results, start=1):
    chunk_id, metadata, content, similarity = row

    if chunk_id in [78, 79]:
        print(f"\nChunk ID: {chunk_id}")
        print(f"Actual rank: {rank}")
        print(f"Page: {metadata.get('page')}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Content:\n{content}")
