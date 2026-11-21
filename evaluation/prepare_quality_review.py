
import json
from pathlib import Path

from app.retrieval.retriever import retrieve_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "generation_evaluation_dataset.json"
)

RESULTS_PATH = (
    PROJECT_ROOT / "evaluation" / "generation_evaluation_results.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation" / "generation_quality_review.json"
)

TOP_K = 6


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def main():

    dataset = load_json(DATASET_PATH)
    previous_results = load_json(RESULTS_PATH)

    dataset_questions = {
        item["id"]: item
        for item in dataset["questions"]
    }

    quality_results = []

    for result in previous_results["results"]:

        question_id = result["id"]
        question_data = dataset_questions[question_id]

        print(f"\nPreparing evidence for {question_id}")

        # Retrieve context without generating another answer.
        chunks = retrieve_documents(
            result["question"],
            top_k=TOP_K
        )

        retrieved_context = []

        for rank, chunk in enumerate(chunks, start=1):

            retrieved_context.append({
                "rank": rank,
                "chunk_id": chunk["id"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "similarity": float(chunk["similarity"]),
                "text": chunk["text"]
            })

        quality_results.append({

            "id": question_id,

            "question": result["question"],

            "type": question_data["type"],

            "expected_answer": question_data["expected_answer"],

            "key_facts": question_data["key_facts"],

            "expected_source": question_data["expected_source"],

            "expected_pages": question_data["expected_pages"],

            # Preserve the previously generated answer.
            "generated_answer": result["generated_answer"],

            "citations": result["citations"],

            # Store the actual evidence available to the generator.
            "retrieved_context": retrieved_context,

            # These fields will be filled during manual review.
            "manual_review": {
                "factual_correctness": None,
                "semantic_faithfulness": None,
                "completeness": None,
                "citation_accuracy": None,
                "review_notes": ""
            }

        })

    output = {
        "dataset": dataset["dataset_name"],
        "top_k": TOP_K,
        "total_questions": len(quality_results),
        "review_status": "PENDING",
        "results": quality_results
    }

    save_json(OUTPUT_PATH, output)

    print("\n" + "=" * 70)
    print("QUALITY REVIEW DATA PREPARATION COMPLETE")
    print("=" * 70)

    print("Questions:", len(quality_results))
    print("Retrieved context saved:", True)
    print("Previous generated answers preserved:", True)
    print("LLM calls made: 0")
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
