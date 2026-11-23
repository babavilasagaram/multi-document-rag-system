
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_dataset_expansion_v1.json"

# Existing judgments must match exactly; otherwise abort.
EXPECTED_EXISTING = {
    "60": 1,
    "61": 1,
    "63": 1,
    "74": 0,
    "80": 0,
    "81": 1,
    "83": 2,
}

# Seven newly approved judgments.
APPROVED_ADDITIONS = {
    "64": 1,
    "70": 2,
    "71": 1,
    "73": 1,
    "82": 1,
    "84": 0,
    "106": 0,
}


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    q02 = next(
        (item for item in data["queries"] if item["qid"] == "q02"),
        None,
    )

    if q02 is None:
        raise ValueError("Q02 not found in dataset.")

    relevance = q02["relevance"]

    # Safety check: never overwrite unexpected existing judgments.
    if relevance != EXPECTED_EXISTING:
        raise ValueError(
            "Existing Q02 judgments do not match expectations.\n"
            f"Expected: {EXPECTED_EXISTING}\n"
            f"Found:    {relevance}\n"
            "No changes made."
        )

    # Create backup before modifying the dataset.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DATASET_PATH.with_name(
        f"{DATASET_PATH.stem}_backup_{timestamp}.json"
    )
    shutil.copy2(DATASET_PATH, backup_path)

    # Add approved judgments.
    relevance.update(APPROVED_ADDITIONS)

    # Record the review decision.
    data.setdefault("review_history", []).append({
        "batch": "Q02 candidate audit",
        "decision": "User approved seven additional Q02 relevance judgments",
        "changes": [
            {
                "qid": "q02",
                "chunk_id": chunk_id,
                "old_grade": "unjudged",
                "new_grade": grade,
            }
            for chunk_id, grade in APPROVED_ADDITIONS.items()
        ],
    })

    with DATASET_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")

    # Reopen and verify the saved result.
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        verified = json.load(file)

    verified_q02 = next(
        item for item in verified["queries"] if item["qid"] == "q02"
    )

    expected_final = {**EXPECTED_EXISTING, **APPROVED_ADDITIONS}

    if verified_q02["relevance"] != expected_final:
        raise RuntimeError("Post-save verification failed.")

    print("Q02 updated successfully.")
    print(f"Backup: {backup_path}")
    print(f"Total Q02 judgments: {len(expected_final)}")
    print("Final judgments:", dict(sorted(
        verified_q02["relevance"].items(),
        key=lambda item: int(item[0])
    )))


if __name__ == "__main__":
    main()
