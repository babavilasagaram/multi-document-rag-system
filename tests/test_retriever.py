
from app.retrieval.retriever import retrieve_multi_query


queries = [
    (
        "Since the Transformer contains no recurrence or convolution, "
        "why do we add positional encodings to the input embeddings?"
    ),
    (
        "How are sine and cosine functions of different frequencies "
        "used to calculate sinusoidal positional encodings?"
    )
]

results = retrieve_multi_query(
    queries,
    top_k_per_query=5,
    final_k=10
)

print("\n========== COMBINED MULTI-QUERY RESULTS ==========")

for rank, chunk in enumerate(results, 1):
    print(f"\nRank: {rank}")
    print(f"Page: {chunk['metadata']['page']}")
    print(f"Similarity: {chunk['similarity']:.4f}")
    print(f"Content: {chunk['text'][:400]}")
