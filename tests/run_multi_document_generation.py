
from app.retrieval.retriever import retrieve_documents
from app.generation.generator import generate_answer


# --------------------------------------------------
# 1. Evaluation Questions
# --------------------------------------------------

test_cases = [
    {
        "question": "How many attention heads does the original Transformer use?",
        "expected_source": "NIPS-2017-attention-is-all-you-need-Paper.pdf"
    },
    {
        "question": "How many items did independent sellers in the US sell in 2023?",
        "expected_source": "amazon_small_business_report_2023.pdf"
    },
    {
        "question": "What should partners do to improve communication after an argument?",
        "expected_source": "amity_relationship_care.pdf"
    },
    {
        "question": "What do group dynamics refer to?",
        "expected_source": "psychology_human_relationships.pdf"
    }
]


# --------------------------------------------------
# 2. Retrieval and Generation
# --------------------------------------------------

for index, test in enumerate(test_cases, start=1):

    question = test["question"]
    expected_source = test["expected_source"]

    print("\n" + "=" * 90)
    print(f"TEST CASE {index}")
    print("=" * 90)

    print("\nQUESTION:")
    print(question)

    # Retrieve relevant chunks
    retrieved_chunks = retrieve_documents(
        question,
        top_k=5
    )

    print("\nRETRIEVED SOURCES:")

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        print(
            f"Rank {rank} | "
            f"Source: {chunk['metadata']['source']} | "
            f"Page: {chunk['metadata']['page']} | "
            f"Similarity: {chunk['similarity']:.4f}"
        )

    # Generate answer
    answer = generate_answer(
        question,
        retrieved_chunks
    )

    print("\nGENERATED ANSWER:")
    print(answer)

    # Basic source presence check
    expected_found = any(
        expected_source in chunk["metadata"]["source"]
        for chunk in retrieved_chunks
    )

    print("\nEXPECTED SOURCE RETRIEVED:", expected_found)

print("\n" + "=" * 90)
print("MULTI-DOCUMENT GENERATION TEST COMPLETED")
print("=" * 90)
