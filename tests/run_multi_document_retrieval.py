
from app.retrieval.retriever import retrieve_documents, retrieve_multi_query


queries = [
    "What should I ask my partner to understand what they need after an argument?",
    "What role does listening play in communication between romantic partners?"
]

for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    results = retrieve_documents(query, top_k=5)

    for rank, chunk in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print("Chunk ID:", chunk["id"])
        print("Source:", chunk["metadata"].get("source"))
        print("Page:", chunk["metadata"].get("page"))
        print("Similarity:", round(chunk["similarity"], 4))
        print("Content:", chunk["text"][:350].replace("\n", " "))


print("\n\nMULTI-QUERY RRF TEST")
print("=" * 80)

results = retrieve_multi_query(
    queries=[
        "How can people improve communication in relationships?",
        "What helps people communicate better with their partners?",
        "How can communication problems between people be addressed?"
    ],
    top_k_per_query=5,
    final_k=5
)

for rank, chunk in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print("Chunk ID:", chunk["id"])
    print("Source:", chunk["metadata"].get("source"))
    print("Similarity:", round(chunk["similarity"], 4))
    print("RRF score:", round(chunk["rrf_score"], 5))
    print("Content:", chunk["text"][:350].replace("\n", " "))
