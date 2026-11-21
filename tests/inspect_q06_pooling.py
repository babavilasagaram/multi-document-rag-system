
from app.retrieval.retriever import retrieve_documents

queries = [
    "How does multi-head attention work?",
    "How are multiple attention heads constructed and combined?",
    "How does the Transformer perform attention across several representation subspaces?",
    "How are queries, keys and values projected into multiple attention heads, and how are the attention operations performed in parallel?"
]

candidates = {}

for query in queries:
    results = retrieve_documents(query, top_k=5)

    for chunk in results:
        chunk_id = chunk["id"]

        if chunk_id not in candidates:
            candidates[chunk_id] = chunk
        elif chunk["similarity"] > candidates[chunk_id]["similarity"]:
            candidates[chunk_id] = chunk

ranked = sorted(
    candidates.values(),
    key=lambda chunk: (-chunk["similarity"], chunk["id"])
)

print("\nQ06 — FOUR-QUERY CANDIDATE POOLING")
print("=" * 65)

for rank, chunk in enumerate(ranked[:10], start=1):
    print(
        f"Rank: {rank:2} | "
        f"ID: {chunk['id']:3} | "
        f"Similarity: {chunk['similarity']:.4f}"
    )
