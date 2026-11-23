
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_dataset_expansion_v1.json"

EXPECTED_EXISTING = {
    "69": 2,
    "70": 2,
    "57": 0,
    "71": 1,
    "72": 0,
}

APPROVED_ADDITIONS = {
    "56": 0,
    "68": 1,
    "73": 0,
    "75": 1,
    "80": 0,
    "102": 0,
}


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    q04 = next(
        (item for item in data["queries"] if item["qid"] == "q04"),
        None,
    )

    if q04 is None:
        raise ValueError("Q04 not found.")

    relevance = q04["relevance"]

    if relevance != EXPECTED_EXISTING:
        raise ValueError(
            "Existing Q04 judgments do not match expectations.\n"
            f"Expected: {EXPECTED_EXISTING}\n"
            f"Found: {relevance}\n"
            "No changes made."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DATASET_PATH.with_name(
        f"{DATASET_PATH.stem}_backup_{timestamp}.json"
    )
    shutil.copy2(DATASET_PATH, backup_path)

    relevance.update(APPROVED_ADDITIONS)

    data.setdefault("review_history", []).append({
        "batch": "Q04 candidate audit",
        "decision": "User approved six additional Q04 relevance judgments",
        "changes": [
            {
                "qid": "q04",
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

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        verified = json.load(file)

    verified_q04 = next(
        item for item in verified["queries"] if item["qid"] == "q04"
    )

    expected_final = {**EXPECTED_EXISTING, **APPROVED_ADDITIONS}

    if verified_q04["relevance"] != expected_final:
        raise RuntimeError("Post-save verification failed.")

    print("Q04 updated successfully.")
    print(f"Backup: {backup_path}")
    print(f"Total Q04 judgments: {len(expected_final)}")
    print("Final judgments:", dict(sorted(
        verified_q04["relevance"].items(),
        key=lambda item: int(item[0])
    )))


if __name__ == "__main__":
    main()
