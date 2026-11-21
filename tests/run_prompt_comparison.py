
import json
from pathlib import Path

from app.retrieval.retriever import retrieve_documents
from app.generation.generator import generate_answer


# --------------------------------------------------
# 1. Evaluation questions
# --------------------------------------------------

TEST_CASES = {
    "G03": "Why does the Transformer use positional encoding?",

    "G04": "How many items did independent sellers in the United States sell in 2023?",

    "G07": "Why is acknowledging a partner's feelings important after an argument?",

    "G09": "What are some helpful ways partners can improve their relationship according to the book?"
}


TOP_K = 6


# --------------------------------------------------
# 2. Run generation tests
# --------------------------------------------------

results = []

for test_id, question in TEST_CASES.items():

    print("\n" + "=" * 70)
    print(f"TEST: {test_id}")
    print(f"QUESTION: {question}")
    print("=" * 70)

    # Retrieve the same top-k configuration
    retrieved_chunks = retrieve_documents(
        question,
        top_k=TOP_K
    )

    print("\nRETRIEVED CHUNKS:")

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        print(
            f"Rank {rank} | "
            f"ID: {chunk['id']} | "
            f"Source: {chunk['metadata']['source']} | "
            f"Page: {chunk['metadata']['page']} | "
            f"Similarity: {chunk['similarity']:.4f}"
        )

    # Generate using the updated prompt
    answer = generate_answer(
        question,
        retrieved_chunks
    )

    print("\nGENERATED ANSWER:")
    print(answer)

    results.append({
        "id": test_id,
        "question": question,
        "top_k": TOP_K,
        "retrieved_chunks": [
            {
                "chunk_id": chunk["id"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "similarity": float(chunk["similarity"])
            }
            for chunk in retrieved_chunks
        ],
        "generated_answer": answer
    })


# --------------------------------------------------
# 3. Save results
# --------------------------------------------------

output_path = Path("evaluation/prompt_comparison_results.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print("\n" + "=" * 70)
print("PROMPT COMPARISON TEST COMPLETED")
print(f"Tests executed: {len(results)}")
print(f"Results saved to: {output_path}")
print("=" * 70)
