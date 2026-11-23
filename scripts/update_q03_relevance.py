
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_dataset_expansion_v1.json"

EXPECTED_EXISTING = {
    "81": 2,
    "82": 1,
    "78": 2,
    "83": 2,
    "63": 1,
    "85": 1,
    "80": 0,
}

APPROVED_ADDITIONS = {
    "65": 0,
    "74": 0,
    "76": 0,
    "86": 0,
    "101": 1,
}


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    q03 = next(
        (item for item in data["queries"] if item["qid"] == "q03"),
        None,
    )

    if q03 is None:
        raise ValueError("Q03 not found.")

    relevance = q03["relevance"]

    if relevance != EXPECTED_EXISTING:
        raise ValueError(
            "Existing Q03 judgments do not match expectations.\n"
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
        "batch": "Q03 candidate audit",
        "decision": "User approved five additional Q03 relevance judgments",
        "changes": [
            {
                "qid": "q03",
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

    verified_q03 = next(
        item for item in verified["queries"] if item["qid"] == "q03"
    )

    expected_final = {**EXPECTED_EXISTING, **APPROVED_ADDITIONS}

    if verified_q03["relevance"] != expected_final:
        raise RuntimeError("Post-save verification failed.")

    print("Q03 updated successfully.")
    print(f"Backup: {backup_path}")
    print(f"Total Q03 judgments: {len(expected_final)}")
    print("Final judgments:", dict(sorted(
        verified_q03["relevance"].items(),
        key=lambda item: int(item[0])
    )))


if __name__ == "__main__":
    main()
