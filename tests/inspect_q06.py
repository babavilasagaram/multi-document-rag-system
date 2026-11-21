
from app.retrieval.retriever import retrieve_documents

query = "How does multi-head attention work?"

results = retrieve_documents(query, top_k=15)

print("\nQ06 — TOP 15 RETRIEVAL RESULTS")
print("=" * 80)

for rank, result in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"Chunk ID: {result['id']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Content: {result['text'][:350]}")
    print("-" * 80)
