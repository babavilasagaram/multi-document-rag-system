
import json
import re
from pathlib import Path

from app.retrieval.retriever import retrieve_documents
from app.generation.generator import generate_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "generation_evaluation_dataset.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation" / "generation_evaluation_results.json"
)

TOP_K = 6


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_citations(answer):
    """
    Extract formatted source citations from the generated answer.
    Expected format:
    [Source: data/example.pdf, Page: 5]
    """

    pattern = (
        r"\[Source:\s*(.*?),\s*Page:\s*(\d+)\]"
    )

    return [
        {
            "source": match[0].strip(),
            "page": int(match[1])
        }
        for match in re.findall(pattern, answer)
    ]


def is_abstention(answer):
    abstention_messages = [
        "I cannot find the answer in the provided documents.",
        "I cannot find relevant information in the documents."
    ]

    return answer.strip().casefold() in [
        message.casefold()
        for message in abstention_messages
    ]


def evaluate_question(question_data):

    question_id = question_data["id"]
    question = question_data["question"]

    print("\n" + "=" * 80)
    print(f"QUESTION ID: {question_id}")
    print("=" * 80)

    print("Question:", question)

    # 1. Retrieve documents
    retrieved_chunks = retrieve_documents(
        question,
        top_k=TOP_K
    )

    retrieved_sources = [
        chunk["metadata"]["source"]
        for chunk in retrieved_chunks
    ]

    print("\nRetrieved Sources:")

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        print(
            f"Rank {rank} | "
            f"Source: {chunk['metadata']['source']} | "
            f"Page: {chunk['metadata']['page']} | "
            f"Similarity: {chunk['similarity']:.4f}"
        )

    # 2. Generate answer
    answer = generate_answer(
        question,
        retrieved_chunks
    )

    print("\nGenerated Answer:")
    print(answer)

    citations = get_citations(answer)

    answerable = question_data["type"] == "answerable"

    # 3. Evaluate retrieval source
    expected_source = question_data["expected_source"]

    if answerable:
        expected_source_retrieved = any(
            Path(source).name == expected_source
            for source in retrieved_sources
        )
    else:
        expected_source_retrieved = None

    # 4. Evaluate citations
    if answerable:
        expected_source_cited = any(
            Path(citation["source"]).name == expected_source
            for citation in citations
        )

        expected_page_cited = any(
            Path(citation["source"]).name == expected_source
            and citation["page"] in question_data["expected_pages"]
            for citation in citations
        )
    else:
        expected_source_cited = None
        expected_page_cited = None

    # 5. Evaluate abstention
    abstained = is_abstention(answer)

    if not answerable:
        abstention_correct = abstained
    else:
        abstention_correct = None

    result = {
        "id": question_id,
        "question": question,
        "type": question_data["type"],
        "generated_answer": answer,
        "expected_answer": question_data["expected_answer"],
        "retrieved_sources": retrieved_sources,
        "citations": citations,
        "expected_source_retrieved": expected_source_retrieved,
        "expected_source_cited": expected_source_cited,
        "expected_page_cited": expected_page_cited,
        "abstained": abstained,
        "abstention_correct": abstention_correct
    }

    print("\nEvaluation:")

    print("Expected Source Retrieved:", expected_source_retrieved)
    print("Expected Source Cited:", expected_source_cited)
    print("Expected Page Cited:", expected_page_cited)
    print("Abstained:", abstained)
    print("Correct Abstention:", abstention_correct)

    return result


def calculate_metrics(results):

    answerable = [
        result for result in results
        if result["type"] == "answerable"
    ]

    unanswerable = [
        result for result in results
        if result["type"] == "unanswerable"
    ]

    def rate(values):
        return (
            sum(values) / len(values)
            if values else 0
        )

    retrieval_rate = rate([
        result["expected_source_retrieved"]
        for result in answerable
    ])

    citation_source_rate = rate([
        result["expected_source_cited"]
        for result in answerable
    ])

    citation_page_rate = rate([
        result["expected_page_cited"]
        for result in answerable
    ])

    abstention_rate = rate([
        result["abstention_correct"]
        for result in unanswerable
    ])

    print("\n" + "=" * 80)
    print("MULTI-DOCUMENT GENERATION EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Total Questions: {len(results)}")
    print(f"Answerable Questions: {len(answerable)}")
    print(f"Unanswerable Questions: {len(unanswerable)}")

    print(f"\nExpected Source Retrieval Rate: {retrieval_rate:.2%}")
    print(f"Expected Source Citation Rate: {citation_source_rate:.2%}")
    print(f"Expected Page Citation Rate: {citation_page_rate:.2%}")
    print(f"Correct Abstention Rate: {abstention_rate:.2%}")

    print("\nAnswer correctness: NOT EVALUATED")
    print("Semantic faithfulness: NOT EVALUATED")

    return {
        "total_questions": len(results),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "expected_source_retrieval_rate": retrieval_rate,
        "expected_source_citation_rate": citation_source_rate,
        "expected_page_citation_rate": citation_page_rate,
        "correct_abstention_rate": abstention_rate,
        "answer_correctness": "NOT EVALUATED",
        "semantic_faithfulness": "NOT EVALUATED"
    }


def main():

    dataset = load_dataset()

    results = []

    for question_data in dataset["questions"]:
        result = evaluate_question(question_data)
        results.append(result)

    metrics = calculate_metrics(results)

    output = {
        "dataset": dataset["dataset_name"],
        "metrics": metrics,
        "results": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print("\nResults saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
