
import json
import shutil
from datetime import datetime
from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASELINE_PATH = Path(
    "evaluation/generation_quality_review.json"
)

EXPERIMENT_PATH = Path(
    "evaluation/completeness_rule_experiment_results.json"
)

OUTPUT_PATH = Path(
    "evaluation/completeness_rule_quality_review.json"
)

EXPECTED_IDS = [f"G{i:02d}" for i in range(1, 17)]

DIMENSIONS = [
    "factual_correctness",
    "semantic_faithfulness",
    "completeness",
    "citation_accuracy"
]


# --------------------------------------------------
# 2. Approved candidate scores
# --------------------------------------------------

approved_scores = {
    "G01": [2, 2, 2, 2],
    "G02": [2, 2, 2, 2],
    "G03": [2, 2, 1, 2],
    "G04": [2, 2, 1, 2],
    "G05": [2, 2, 2, 2],
    "G06": [2, 2, 2, 2],
    "G07": [2, 1, 2, 1],
    "G08": [2, 2, 2, 2],
    "G09": [2, 1, 1, 2],
    "G10": [2, 2, 2, 2],
    "G11": [2, 2, 2, 2],
    "G12": [2, 2, 2, 2],
}

review_notes = {
    "G01": "Correct, complete answer with appropriate supporting citation.",
    "G02": "Correct explanation of scaled dot-product attention. Duplicate citation is redundant but valid.",
    "G03": "Explains positional information and sequence order, but omits the sine/cosine implementation detail present in the retrieved context.",
    "G04": "States more than 4.5 billion items but omits the approximately 8,600 items per minute detail present in the retrieved context.",
    "G05": "Correct, complete answer with appropriate supporting citation.",
    "G06": "Correct, complete answer with appropriate supporting citation.",
    "G07": "Core explanation is supported, but the claim about promoting trust is not clearly established by the retrieved context; citation precision is weakened.",
    "G08": "Correct, complete explanation with appropriate supporting citation.",
    "G09": "Several recommendations are supported, but communicating needs and explicit apology are not fully covered as distinct actions; some repair wording is interpretive.",
    "G10": "Correct definition with appropriate supporting citation.",
    "G11": "Correct distinction between cooperation and competition with appropriate supporting citation.",
    "G12": "Correctly identifies all three types of ingroup cooperation with appropriate supporting citation.",
}


# --------------------------------------------------
# 3. Validate source files
# --------------------------------------------------

for path in [BASELINE_PATH, EXPERIMENT_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"Refusing to overwrite: {OUTPUT_PATH}"
    )

baseline = json.loads(
    BASELINE_PATH.read_text(encoding="utf-8")
)

experiment = json.loads(
    EXPERIMENT_PATH.read_text(encoding="utf-8")
)

baseline_records = baseline["results"]
candidate_records = experiment["results"]

if len(baseline_records) != 16:
    raise ValueError("Baseline review must contain 16 records.")

if len(candidate_records) != 16:
    raise ValueError("Candidate experiment must contain 16 records.")

baseline_by_id = {r["id"]: r for r in baseline_records}
candidate_by_id = {r["id"]: r for r in candidate_records}

if set(baseline_by_id) != set(EXPECTED_IDS):
    raise ValueError("Unexpected IDs in baseline review.")

if set(candidate_by_id) != set(EXPECTED_IDS):
    raise ValueError("Unexpected IDs in candidate experiment.")

if len(baseline_by_id) != 16 or len(candidate_by_id) != 16:
    raise ValueError("Duplicate question IDs detected.")


# --------------------------------------------------
# 4. Verify records and preserve baseline
# --------------------------------------------------

for qid in EXPECTED_IDS:
    base = baseline_by_id[qid]
    candidate = candidate_by_id[qid]

    if not candidate.get("frozen_chunks"):
        raise ValueError(f"{qid}: Missing frozen context.")

    if not candidate.get("original_prompt_answer"):
        raise ValueError(f"{qid}: Missing baseline answer.")

    if not candidate.get("candidate_prompt_answer"):
        raise ValueError(f"{qid}: Missing candidate answer.")

    if base.get("question") != candidate.get("question"):
        raise ValueError(f"{qid}: Question mismatch.")

    if qid in approved_scores:
        if base.get("type") != "answerable":
            raise ValueError(f"{qid}: Expected answerable question.")

    else:
        if base.get("type") != "unanswerable":
            raise ValueError(f"{qid}: Expected unanswerable question.")


# --------------------------------------------------
# 5. Build separate candidate review
# --------------------------------------------------

results = []

for qid in EXPECTED_IDS:
    base = baseline_by_id[qid]
    candidate = candidate_by_id[qid]

    item = {
        "id": qid,
        "type": base["type"],
        "question": candidate["question"],
        "original_prompt_answer": candidate["original_prompt_answer"],
        "candidate_prompt_answer": candidate["candidate_prompt_answer"],
    }

    if qid in approved_scores:
        values = approved_scores[qid]

        manual_review = {
            dimension: score
            for dimension, score in zip(DIMENSIONS, values)
        }

        manual_review["total_score"] = sum(values)
        manual_review["notes"] = review_notes[qid]
        manual_review["review_status"] = "COMPLETED"

        item["candidate_manual_review"] = manual_review

    else:
        item["abstention_check"] = {
            "candidate_correct": (
                candidate["candidate_prompt_answer"].strip()
                == "I cannot find the answer in the provided documents."
            )
        }

    results.append(item)


# --------------------------------------------------
# 6. Calculate candidate aggregate
# --------------------------------------------------

dimension_totals = {
    dimension: sum(
        approved_scores[qid][index]
        for qid in approved_scores
    )
    for index, dimension in enumerate(DIMENSIONS)
}

dimension_maximum = 12 * 2

total_score = sum(dimension_totals.values())
total_maximum = 12 * 8

correct_abstentions = sum(
    item.get("abstention_check", {}).get(
        "candidate_correct", False
    )
    for item in results
    if item["type"] == "unanswerable"
)

summary = {
    "total_questions": 16,
    "answerable_questions": 12,
    "unanswerable_questions": 4,
    "candidate_manual_quality_score": total_score,
    "candidate_manual_quality_maximum": total_maximum,
    "candidate_manual_quality_percentage": round(
        total_score / total_maximum * 100, 2
    ),
    "dimension_scores": {
        dimension: {
            "score": score,
            "maximum": dimension_maximum,
            "percentage": round(
                score / dimension_maximum * 100, 2
            )
        }
        for dimension, score in dimension_totals.items()
    },
    "candidate_correct_abstentions": (
        f"{correct_abstentions}/4"
    )
}


# --------------------------------------------------
# 7. Create output and backup
# --------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

backup_path = BASELINE_PATH.with_name(
    f"{BASELINE_PATH.stem}_backup_{timestamp}.json"
)

if backup_path.exists():
    raise FileExistsError(
        f"Backup already exists: {backup_path}"
    )

# Backup baseline; do not modify it.
shutil.copy2(BASELINE_PATH, backup_path)

output_data = {
    "experiment": "Single completeness rule manual quality review",
    "source_experiment": str(EXPERIMENT_PATH),
    "baseline_review": str(BASELINE_PATH),
    "review_status": "COMPLETED",
    "scoring_rubric": baseline.get("scoring_rubric"),
    "summary": summary,
    "results": results
}

with OUTPUT_PATH.open("x", encoding="utf-8") as file:
    json.dump(
        output_data,
        file,
        indent=4,
        ensure_ascii=False
    )


# --------------------------------------------------
# 8. Post-save verification
# --------------------------------------------------

saved = json.loads(
    OUTPUT_PATH.read_text(encoding="utf-8")
)

assert len(saved["results"]) == 16
assert saved["summary"]["candidate_manual_quality_score"] == 91
assert saved["summary"]["candidate_manual_quality_maximum"] == 96
assert saved["summary"]["candidate_correct_abstentions"] == "4/4"

assert BASELINE_PATH.exists()
assert backup_path.exists()

print("\nCOMPLETENESS RULE REVIEW UPDATE COMPLETED")
print("Candidate score:", total_score, "/", total_maximum)
print("Candidate quality:", summary["candidate_manual_quality_percentage"], "%")
print("Correct abstentions:", correct_abstentions, "/4")
print("Backup:", backup_path)
print("New review:", OUTPUT_PATH)
print("Original baseline review: UNMODIFIED")
print("Post-save verification: PASSED")
