
from app.retrieval.retriever import (
    retrieve_documents,
    retrieve_multi_query
)

queries = [
    "Why does the Transformer need positional information?",

    "Since the Transformer contains no recurrence and no convolution, "
    "why must information about the order of the sequence be injected?",

    "Why must the model make use of the order of the sequence?",

    "How does the Transformer represent the relative or absolute position "
    "of tokens in a sequence?"
]

# Candidate pooling
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

candidate_results = sorted(
    candidates.values(),
    key=lambda x: (-x["similarity"], x["id"])
)

# Reciprocal Rank Fusion
rrf_results = retrieve_multi_query(
    queries,
    top_k_per_query=5,
    final_k=5
)

candidate_results = candidate_results[:5]

baseline_results = retrieve_documents(
    queries[0],
    top_k=5
)

# Display comparison
for name, results in [
    ("BASELINE", baseline_results),
    ("CANDIDATE POOLING", candidate_results),
    ("RECIPROCAL RANK FUSION", rrf_results)
]:
    print("\n" + "=" * 70)
    print(name)

    for rank, result in enumerate(results, start=1):
        if "rrf_score" in result:
            score_info = (
                f"RRF Score: {result['rrf_score']:.6f} | "
                f"Similarity: {result['similarity']:.4f}"
            )
        else:
            score_info = f"Similarity: {result['similarity']:.4f}"

        print(
            f"Rank: {rank:2} | "
            f"ID: {result['id']} | "
            f"{score_info}"
        )