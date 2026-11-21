from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_documents
from app.embeddings.embedder import embed_documents, embed_query


documents = load_pdf(
    "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
)

chunks = chunk_documents(documents)

embedded_chunks = embed_documents(chunks)

print("Total embedded chunks:", len(embedded_chunks))

print("\nFirst embedding:")
print(embedded_chunks[0]["embedding"][:10])

print("\nEmbedding dimensions:")
print(len(embedded_chunks[0]["embedding"]))

query_vector = embed_query("What is the Transformer architecture?")

print("\nQuery embedding dimensions:", len(query_vector))