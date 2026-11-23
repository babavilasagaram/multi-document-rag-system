
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation_dataset_expansion_v1.json"
)

# Exact existing judgments captured during batch inspection.

EXPECTED_EXISTING = {
    "q05": {
        "71": 2,
        "70": 1,
        "69": 1,
        "57": 0,
        "68": 0,
        "72": 1,
    },
    "q06": {
        "72": 2,
        "73": 2,
        "71": 1,
        "69": 1,
        "66": 1,
        "74": 1,
        "63": 1,
        "68": 1,
        "114": 1,
    },
    "q07": {
        "73": 2,
        "72": 1,
        "63": 2,
        "74": 0,
        "71": 1,
        "69": 0,
        "67": 0,
    },
    "q08": {
        "74": 2,
        "75": 2,
        "67": 1,
        "68": 1,
        "66": 1,
        "102": 0,
        "114": 0,
    },
    "q09": {
        "68": 2,
        "75": 2,
        "67": 2,
        "74": 0,
        "81": 0,
    },
}


# User-approved additions only.
APPROVED_ADDITIONS = {
    "q05": {
        "62": 0,
        "73": 0,
        "74": 0,
        "80": 0,
        "81": 0,
        "88": 0,
        "96": 0,
        "102": 0,
        "428": 0,
    },
    "q06": {
        "86": 1,
        "102": 0,
        "115": 1,
        "116": 1,
    },
    "q07": {
        "68": 0,
        "70": 0,
        "80": 0,
        "83": 0,
        "114": 1,
        "115": 1,
        "116": 1,
    },
    "q08": {
        "56": 0,
        "60": 0,
        "76": 0,
        "80": 0,
        "81": 0,
        "90": 0,
        "99": 0,
    },
    "q09": {
        "60": 0,
        "63": 0,
        "66": 0,
        "80": 0,
        "86": 0,
        "101": 0,
        "114": 0,
    },
}


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    questions = {
        item["qid"]: item
        for item in data["queries"]
    }

    # Validate all questions before making any modification.
    for qid in EXPECTED_EXISTING:
        if qid not in questions:
            raise ValueError(f"{qid} not found in dataset.")

        actual = questions[qid]["relevance"]
        expected = EXPECTED_EXISTING[qid]

        if actual != expected:
            raise ValueError(
                f"Existing judgments mismatch for {qid}.\n"
                f"Expected: {expected}\n"
                f"Found: {actual}\n"
                "No changes made."
            )

    # Ensure additions do not overwrite existing judgments.
    for qid, additions in APPROVED_ADDITIONS.items():
        overlap = (
            set(additions) & set(questions[qid]["relevance"])
        )
        if overlap:
            raise ValueError(
                f"Unexpected existing labels in {qid}: {overlap}\n"
                "No changes made."
            )

    # Create one backup before modifying anything.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DATASET_PATH.with_name(
        f"{DATASET_PATH.stem}_backup_{timestamp}.json"
    )
    shutil.copy2(DATASET_PATH, backup_path)

    # Apply approved additions and record review history.
    for qid, additions in APPROVED_ADDITIONS.items():
        questions[qid]["relevance"].update(additions)

        data.setdefault("review_history", []).append({
            "batch": "Q05-Q09 candidate audit",
            "decision": (
                "User approved all proposed relevance judgments."
            ),
            "changes": [
                {
                    "qid": qid,
                    "chunk_id": chunk_id,
                    "old_grade": "unjudged",
                    "new_grade": grade,
                }
                for chunk_id, grade in additions.items()
            ],
        })

    # Save.
    with DATASET_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")

    # Reopen and verify every question.
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        verified = json.load(file)

    verified_questions = {
        item["qid"]: item
        for item in verified["queries"]
    }

    print("BATCH UPDATE COMPLETED")
    print(f"Backup: {backup_path}")

    total_added = 0

    for qid in APPROVED_ADDITIONS:
        expected_final = {
            **EXPECTED_EXISTING[qid],
            **APPROVED_ADDITIONS[qid],
        }

        actual_final = verified_questions[qid]["relevance"]

        if actual_final != expected_final:
            raise RuntimeError(
                f"Post-save verification failed for {qid}."
            )

        added = len(APPROVED_ADDITIONS[qid])
        total_added += added

        print(
            f"{qid.upper()}: VERIFIED | "
            f"Existing: {len(EXPECTED_EXISTING[qid])} | "
            f"Added: {added} | "
            f"Total: {len(actual_final)}"
        )

    print(f"\nTotal new judgments: {total_added}")
    print("All five questions passed verification.")


if __name__ == "__main__":
    main()
