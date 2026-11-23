from app.retrieval.retriever import retrieve_documents

question = "Why does the Transformer scale the dot products by the square root of the key dimension?"

results = retrieve_documents(question, top_k=5)

for i, result in enumerate(results, 1):
    print(f"\n{'=' * 70}")
    print(f"RESULT {i}")
    print(f"Similarity: {result.get('similarity')}")
    print(f"Metadata: {result.get('metadata')}")
    print(f"Content:\n{result.get('content')}")