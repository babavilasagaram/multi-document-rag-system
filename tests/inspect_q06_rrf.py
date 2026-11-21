
from app.retrieval.retriever import retrieve_documents

queries = [
    "How does multi-head attention work?",
    "How are multiple attention heads constructed and combined?",
    "How does the Transformer perform attention across several representation subspaces?",
    "How are queries, keys and values projected into multiple attention heads, and how are the attention operations performed in parallel?"
]

target_ids = {71, 72, 73}

print("\nQ06 — INDIVIDUAL QUERY CONTRIBUTIONS")
print("=" * 65)

for query_number, query in enumerate(queries, start=1):
    results = retrieve_documents(query, top_k=5)

    print(f"\nQuery {query_number}: {query}")

    for rank, chunk in enumerate(results, start=1):
        if chunk["id"] in target_ids:
            print(
                f"  Chunk {chunk['id']} | "
                f"Rank: {rank} | "
                f"Similarity: {chunk['similarity']:.4f} | "
                f"RRF contribution: {1 / (60 + rank):.5f}"
            )
