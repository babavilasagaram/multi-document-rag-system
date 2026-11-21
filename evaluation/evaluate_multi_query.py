
import json

from app.retrieval.retriever import retrieve_multi_query


TOP_K_PER_QUERY = 10
FINAL_K = 10


def build_queries(question):
    return [
        question,
        f"What is the purpose and motivation behind this concept: {question}",
        f"Explain the mechanism, implementation, and important details related to: {question}"
    ]


with open("evaluation/test_questions.json", "r", encoding="utf-8") as file:
    test_questions = json.load(file)


total_recall = 0
total_hit = 0
evaluated_questions = 0


for item in test_questions:

    if item["category"] == "out_of_scope":
        continue

    question = item["question"]
    expected_evidence = item.get("expected_evidence", [])

    if not expected_evidence:
        continue

    queries = build_queries(question)

    retrieved_chunks = retrieve_multi_query(
        queries,
        top_k_per_query=TOP_K_PER_QUERY,
        final_k=FINAL_K
    )

    retrieved_evidence = set()

    for rank, chunk in enumerate(retrieved_chunks, 1):

        content = "".join(chunk["text"].casefold().split())

        matched = {
            phrase
            for phrase in expected_evidence
            if "".join(phrase.casefold().split()) in content
        }

        retrieved_evidence.update(matched)

        print(f"\nRank: {rank}")
        print(f"Page: {chunk['metadata']['page']}")
        print(f"Similarity: {chunk['similarity']:.4f}")
        print(f"RRF Score: {chunk['rrf_score']:.6f}")
        print(f"Matched evidence: {matched}")

    recall = len(retrieved_evidence) / len(expected_evidence)
    hit = len(retrieved_evidence) > 0

    total_recall += recall
    total_hit += int(hit)
    evaluated_questions += 1

    print(f"\nQuestion {item['id']}: {question}")
    print(f"Evidence Recall@{FINAL_K}: {recall:.2%}")
    print(f"Hit@{FINAL_K}: {hit}")
    print("-" * 60)


if evaluated_questions:

    print("\n========== RRF MULTI-QUERY EVALUATION ==========")

    print(f"Questions evaluated: {evaluated_questions}")
    print(
        f"Mean Evidence Recall@{FINAL_K}: "
        f"{total_recall / evaluated_questions:.2%}"
    )
    print(
        f"Hit@{FINAL_K}: "
        f"{total_hit / evaluated_questions:.2%}"
    )
