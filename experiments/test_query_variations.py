from app.retrieval.retriever import retrieve_multi_query

queries = [
    "Why does Transformer need positional encoding?",
    "How does Transformer represent the order of tokens?",
    "How does Transformer capture sequence order without recurrence?"
]

results = retrieve_multi_query(
    queries,
    top_k_per_query=5,
    final_k=5
)

for rank, chunk in enumerate(results, start=1):
    print(
        rank,
        chunk["id"],
        round(chunk["rrf_score"], 5),
        chunk["metadata"]
    )