
from app.retrieval.retriever import retrieve_documents

query = "Why does the Transformer use positional encoding?"
results = retrieve_documents(query, top_k=10)

for rank, chunk in enumerate(results, start=1):
    print(
        rank,
        chunk["id"],
        chunk["metadata"]["page"],
        round(chunk["similarity"], 4),
        chunk["text"][:250]
    )