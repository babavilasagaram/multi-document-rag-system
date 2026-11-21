
from app.retrieval.retriever import retrieve_documents

queries = [
    "Why does the Transformer need positional information?",
    "Since the Transformer contains no recurrence and no convolution, "
    "why must information about the order of the sequence be injected?",
    "Why must the model make use of the order of the sequence?",
    "How does the Transformer represent the relative or absolute position "
    "of tokens in a sequence?"
]

candidates = {}

for query in queries:

    results = retrieve_documents(query, top_k=5)

    for result in results:

        chunk_id = result["id"]

        if chunk_id not in candidates:
            candidates[chunk_id] = result.copy()
        else:
            candidates[chunk_id]["similarity"] = max(
                candidates[chunk_id]["similarity"],
                result["similarity"]
            )

ranked_candidates = sorted(
    candidates.values(),
    key=lambda x: (-x["similarity"], x["id"])
)

print("\nCANDIDATE POOLING RESULTS")

for rank, result in enumerate(ranked_candidates, start=1):
    print(
        f"Rank: {rank:2} | "
        f"ID: {result['id']} | "
        f"Similarity: {result['similarity']:.4f}"
    )
