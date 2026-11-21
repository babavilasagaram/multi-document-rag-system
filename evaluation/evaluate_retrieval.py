
import re
import json

from app.retrieval.retriever import retrieve_documents


TOP_K = 10


def normalize_text(text):
    """Normalize text to handle whitespace differences in PDF extraction."""
    return re.sub(r"\s+", "", text.casefold())


with open("evaluation/test_questions.json", "r", encoding="utf-8") as file:
    test_questions = json.load(file)


total_precision = 0
total_recall = 0
total_mrr = 0
total_hit = 0
evaluated_questions = 0


for item in test_questions:

    if item["category"] == "out_of_scope":
        continue

    question = item["question"]
    expected_evidence = item.get("expected_evidence", [])

    if not expected_evidence:
        print(f"Skipping Q{item['id']}: missing expected_evidence")
        continue

    retrieved_chunks = retrieve_documents(
        question,
        top_k=TOP_K
    )

    retrieved_evidence = set()
    relevant_ranks = []

    for rank, chunk in enumerate(retrieved_chunks, 1):

        content = normalize_text(chunk["text"])

        matched_evidence = {
            phrase
            for phrase in expected_evidence
            if normalize_text(phrase) in content
        }

        if matched_evidence:
            relevant_ranks.append(rank)
            retrieved_evidence.update(matched_evidence)

        print(f"\nRank: {rank}")
        print(f"Page: {chunk['metadata']['page']}")
        print(f"Similarity: {chunk['similarity']:.4f}")
        print(f"Matched evidence: {matched_evidence}")
        print(f"Content:\n{chunk['text']}")
        print("-" * 60)

    relevant_chunks = len(relevant_ranks)

    precision = (
        relevant_chunks / len(retrieved_chunks)
        if retrieved_chunks else 0
    )

    recall = len(retrieved_evidence) / len(expected_evidence)

    hit = relevant_chunks > 0

    mrr = (
        1 / relevant_ranks[0]
        if relevant_ranks else 0
    )

    total_precision += precision
    total_recall += recall
    total_mrr += mrr
    total_hit += int(hit)
    evaluated_questions += 1

    print(f"\nQuestion {item['id']}: {question}")
    print(f"Expected evidence: {len(expected_evidence)}")
    print(f"Retrieved evidence: {len(retrieved_evidence)}")
    print(f"Relevant chunks: {relevant_chunks}")
    print(f"Precision@{TOP_K}: {precision:.2f}")
    print(f"Evidence Recall@{TOP_K}: {recall:.2f}")
    print(f"MRR@{TOP_K}: {mrr:.4f}")
    print(f"Hit@{TOP_K}: {hit}")
    print("-" * 60)


if evaluated_questions:

    print("\n========== RETRIEVAL EVALUATION ==========")

    print(f"Questions evaluated: {evaluated_questions}")

    print(
        f"Mean Precision@{TOP_K}: "
        f"{total_precision / evaluated_questions:.2%}"
    )

    print(
        f"Mean Evidence Recall@{TOP_K}: "
        f"{total_recall / evaluated_questions:.2%}"
    )

    print(
        f"Mean MRR@{TOP_K}: "
        f"{total_mrr / evaluated_questions:.4f}"
    )

    print(
        f"Hit@{TOP_K}: "
        f"{total_hit / evaluated_questions:.2%}"
    )
