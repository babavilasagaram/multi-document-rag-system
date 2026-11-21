
from app.retrieval.retriever import retrieve_documents
from app.generation.generator import generate_answer

query = "How many attention heads does the original Transformer use?"


# Step 1: Retrieve relevant chunks
retrieved_chunks = retrieve_documents(query, top_k=3)


# Step 2: Generate grounded answer
answer = generate_answer(query, retrieved_chunks)


# Step 3: Display output
print("\nUSER QUESTION:")
print(query)

print("\nGENERATED ANSWER:")
print(answer)
