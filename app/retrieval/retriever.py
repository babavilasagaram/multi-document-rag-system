
from app.database.connection import get_connection
from app.embeddings.embedder import embed_query


def retrieve_documents(query, top_k=3):

    query_embedding = embed_query(query)
    vector = "[" + ",".join(map(str, query_embedding)) + "]"

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                ORDER BY embedding <=> %s::vector, id
                LIMIT %s;
                """,
                (vector, vector, top_k)
            )

            results = cursor.fetchall()

    return [
        {
            "id": row[0],
            "text": row[1],
            "metadata": row[2],
            "similarity": float(row[3])
        }
        for row in results
    ]




def retrieve_multi_query(
    queries,
    top_k_per_query=5,
    final_k=10,
    rrf_k=60
):
    # Basic input validation
    if not queries:
        raise ValueError("At least one query is required.")

    if top_k_per_query <= 0 or final_k <= 0:
        raise ValueError("top_k_per_query and final_k must be positive.")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    # Remove duplicate queries while preserving order
    unique_queries = list(dict.fromkeys(
        query.strip() for query in queries if query.strip()
    ))

    if not unique_queries:
        raise ValueError("No valid queries provided.")

    fused_chunks = {}

    for query in unique_queries:

        results = retrieve_documents(
            query,
            top_k=top_k_per_query
        )

        for rank, chunk in enumerate(results, start=1):

            # Use database ID as unique chunk identity
            key = chunk["id"]

            if key not in fused_chunks:
                fused_chunks[key] = {
                    **chunk,
                    "rrf_score": 0.0,
                    "best_rank": rank
                }

            fused_chunks[key]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

            fused_chunks[key]["best_rank"] = min(
                fused_chunks[key]["best_rank"],
                rank
            )

            fused_chunks[key]["similarity"] = max(
                fused_chunks[key]["similarity"],
                chunk["similarity"]
            )

    ranked_chunks = sorted(
        fused_chunks.values(),
        key=lambda x: (
            -x["rrf_score"],
            x["best_rank"],
            -x["similarity"],
            x["id"]
        )
    )

    return ranked_chunks[:final_k]
