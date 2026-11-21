
from app.retrieval.retriever import retrieve_documents
from app.generation.generator import generate_answer

query = "Why does the Transformer use positional encoding?"

for top_k in [5, 6, 8, 10]:

    print("\n" + "=" * 70)
    print("TOP_K:", top_k)
    print("=" * 70)

    chunks = retrieve_documents(query, top_k=top_k)

    print("\nRetrieved chunk IDs:")

    for chunk in chunks:
        print(
            chunk["id"],
            chunk["metadata"]["page"],
            round(chunk["similarity"], 4)
        )

    answer = generate_answer(query, chunks)

    print("\nANSWER:")
    print(answer)
