
from app.retrieval.retriever import retrieve_documents

query = (
    "What are some helpful ways partners "
    "can improve their relationship according to the book?"
)

results = retrieve_documents(query, top_k=10)

print("=" * 90)
print("G09 TOP-10 RETRIEVAL INSPECTION")
print("=" * 90)

for rank, chunk in enumerate(results, start=1):

    metadata = chunk.get("metadata") or {}

    print(f"\nRANK: {rank}")
    print(f"ID: {chunk.get('id')}")
    print(f"SOURCE: {metadata.get('source')}")
    print(f"PAGE: {metadata.get('page')}")
    print(f"SIMILARITY: {chunk.get('similarity')}")

    print("\nCONTENT:")
    print(chunk.get("text", ""))

    print("-" * 90)
