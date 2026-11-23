
import json
from pathlib import Path


INPUT_PATH = Path(
    "evaluation/full_prompt_comparison_results.json"
)

OUTPUT_PATH = Path(
    "evaluation/full_prompt_quality_review.json"
)


# --------------------------------------------------
# 1. Load comparison results
# --------------------------------------------------

with INPUT_PATH.open("r", encoding="utf-8") as file:
    experiment = json.load(file)

results = experiment["results"]


# --------------------------------------------------
# 2. Define manual review structure
# --------------------------------------------------

def empty_review():
    return {
        "factual_correctness": None,
        "semantic_faithfulness": None,
        "completeness": None,
        "citation_accuracy": None,
        "total_score": None,
        "notes": "",
        "review_status": "PENDING"
    }


review_results = []

for item in results:

    original_answer = item.get("original_prompt_answer")
    coverage_answer = item.get("coverage_prompt_answer")

    if original_answer is None or coverage_answer is None:
        raise ValueError(
            f"Incomplete answers for {item['id']}"
        )

    record = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "expected_answer": item.get("expected_answer"),
        "key_facts": item.get("key_facts"),
        "expected_source": item.get("expected_source"),
        "expected_pages": item.get("expected_pages"),

        "original_prompt": {
            "answer": original_answer,
            "manual_review": empty_review()
        },

        "coverage_prompt": {
            "answer": coverage_answer,
            "manual_review": empty_review()
        }
    }

    if item["type"] == "answerable":
        record["frozen_chunks"] = item["frozen_chunks"]

    else:
        abstention_message = (
            "I cannot find the answer in the provided documents."
        )

        record["abstention_check"] = {
            "original_correct": (
                original_answer.strip() == abstention_message
            ),
            "coverage_correct": (
                coverage_answer.strip() == abstention_message
            )
        }

    review_results.append(record)


# --------------------------------------------------
# 3. Summary
# --------------------------------------------------

answerable_count = sum(
    item["type"] == "answerable"
    for item in review_results
)

unanswerable_count = sum(
    item["type"] == "unanswerable"
    for item in review_results
)

correct_original_abstentions = sum(
    item.get("abstention_check", {}).get(
        "original_correct", False
    )
    for item in review_results
)

correct_coverage_abstentions = sum(
    item.get("abstention_check", {}).get(
        "coverage_correct", False
    )
    for item in review_results
)


# --------------------------------------------------
# 4. Save review file
# --------------------------------------------------

review_data = {
    "experiment": "Full paired manual prompt quality review",

    "source_experiment": (
        "evaluation/full_prompt_comparison_results.json"
    ),

    "review_status": "PENDING",

    "scoring_rubric": {
        "factual_correctness": {
            "0": "Incorrect or substantially wrong",
            "1": "Partially correct",
            "2": "Correct"
        },

        "semantic_faithfulness": {
            "0": "Contains unsupported claims",
            "1": "Mostly supported, with some unsupported details",
            "2": "Claims are supported by retrieved context"
        },

        "completeness": {
            "0": "Major relevant information is missing",
            "1": "Some relevant information is missing",
            "2": "Important relevant information is covered"
        },

        "citation_accuracy": {
            "0": "Citations are missing, invalid, or unsupported",
            "1": "Some citations are appropriate, others are weak",
            "2": "Citations support their associated claims"
        },

        "total_maximum_per_answer": 8
    },

    "summary": {
        "total_questions": len(review_results),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "original_correct_abstentions": (
            correct_original_abstentions
        ),
        "coverage_correct_abstentions": (
            correct_coverage_abstentions
        ),
        "maximum_answerable_score_per_prompt": (
            answerable_count * 8
        )
    },

    "results": review_results
}


with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        review_data,
        file,
        indent=4,
        ensure_ascii=False
    )


print("=" * 70)
print("PAIRED QUALITY REVIEW PREPARED")
print("=" * 70)

print("Total questions:", len(review_results))
print("Answerable:", answerable_count)
print("Unanswerable:", unanswerable_count)

print(
    "Original abstentions:",
    correct_original_abstentions,
    "/",
    unanswerable_count
)

print(
    "Coverage abstentions:",
    correct_coverage_abstentions,
    "/",
    unanswerable_count
)

print("Maximum score per prompt:", answerable_count * 8)

print("Review status: PENDING")
print("Saved to:", OUTPUT_PATH)

print("=" * 70)
