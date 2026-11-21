
from app.retrieval.retriever import retrieve_documents


queries = [
    "How does multi-head attention work?",

    "How are multiple attention heads constructed and combined?",

    "How does the Transformer perform attention across several representation subspaces?",

    "How are queries, keys and values projected into multiple attention heads, and how are the attention operations performed in parallel?"
]


for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    results = retrieve_documents(query, top_k=15)

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank: {rank:2} | "
            f"ID: {result['id']:3} | "
            f"Similarity: {result['similarity']:.4f}"
        )
