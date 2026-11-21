from app.retrieval.retriever import retrieve_documents

queries = [
    "Why does the Transformer need positional information?",
    "Why is positional encoding added to input embeddings?",
    "How does the Transformer know the position of each token?"
]

for query in queries:

    results = retrieve_documents(query, top_k=20)

    print("\n" + "=" * 70)
    print("QUERY:", query)

    for rank, result in enumerate(results, start=1):

        print(
            f"Rank: {rank:2} | "
            f"ID: {result['id']:3} | "
            f"Similarity: {result['similarity']:.4f}"
        )