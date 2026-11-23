
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_dataset_expansion_v1.json"

EXPECTED_EXISTING = {
    "q10": {
        "76": 2, "66": 1, "75": 1, "82": 0, "68": 0, "74": 0
    },
    "q11": {
        "77": 2, "65": 0, "76": 0, "79": 0, "87": 0, "96": 0
    },
    "q12": {
        "78": 2, "79": 2, "65": 0, "66": 0,
        "74": 0, "90": 0, "93": 0, "94": 0
    },
    "q13": {
        "80": 2, "79": 1, "68": 0,
        "74": 0, "75": 0, "297": 0
    },
    "q14": {
        "79": 2, "80": 2, "59": 0,
        "65": 0, "68": 0, "75": 0
    },
}

APPROVED_ADDITIONS = {
    "q10": {
        "60": 0, "65": 0, "67": 0, "70": 0,
        "78": 0, "81": 0, "83": 0, "85": 0
    },
    "q11": {
        "59": 0, "67": 0, "68": 0, "74": 0, "86": 0,
        "90": 0, "93": 0, "97": 0, "98": 0, "112": 0
    },
    "q12": {
        "56": 0, "61": 0, "62": 0, "68": 0,
        "75": 0, "76": 0, "89": 0, "102": 0
    },
    "q13": {
        "62": 0, "65": 0, "76": 0, "89": 0,
        "94": 0, "96": 1, "373": 0
    },
    "q14": {
        "60": 0, "74": 0, "76": 0, "77": 0,
        "78": 1, "81": 0, "96": 0
    },
}


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    queries = {
        item["qid"]: item
        for item in dataset["queries"]
    }

    # 1. Verify all target questions exist.
    for qid in EXPECTED_EXISTING:
        if qid not in queries:
            raise ValueError(f"{qid} is missing from the dataset.")

    # 2. Verify exact existing relevance dictionaries.
    for qid, expected in EXPECTED_EXISTING.items():
        actual = queries[qid]["relevance"]

        if actual != expected:
            raise ValueError(
                f"\nSAFETY CHECK FAILED for {qid}.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}\n"
                "Dataset was not modified."
            )

    # 3. Ensure no approved addition overwrites an existing grade.
    for qid, additions in APPROVED_ADDITIONS.items():
        existing_ids = set(queries[qid]["relevance"])
        overlap = existing_ids.intersection(additions)

        if overlap:
            raise ValueError(
                f"{qid}: addition IDs already exist: {sorted(overlap)}"
            )

    # 4. Validate relevance grades.
    for qid, additions in APPROVED_ADDITIONS.items():
        for chunk_id, grade in additions.items():
            if grade not in (0, 1, 2):
                raise ValueError(
                    f"Invalid grade {grade} for {qid}, chunk {chunk_id}"
                )

    # 5. Create a timestamped backup before modifying anything.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DATASET_PATH.with_name(
        f"evaluation_dataset_expansion_v1_backup_{timestamp}.json"
    )

    if backup_path.exists():
        raise FileExistsError(f"Backup already exists: {backup_path}")

    shutil.copy2(DATASET_PATH, backup_path)

    # 6. Apply approved additions.
    for qid, additions in APPROVED_ADDITIONS.items():
        queries[qid]["relevance"].update(additions)

    # 7. Save the updated dataset.
    with DATASET_PATH.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, ensure_ascii=False)
        file.write("\n")

    # 8. Reopen and verify the saved file.
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        saved_dataset = json.load(file)

    saved_queries = {
        item["qid"]: item
        for item in saved_dataset["queries"]
    }

    total_added = 0

    for qid, additions in APPROVED_ADDITIONS.items():
        final_relevance = saved_queries[qid]["relevance"]

        for chunk_id, grade in additions.items():
            if final_relevance.get(chunk_id) != grade:
                raise RuntimeError(
                    f"Verification failed: {qid}, chunk {chunk_id}"
                )

        expected_final = EXPECTED_EXISTING[qid] | additions

        if final_relevance != expected_final:
            raise RuntimeError(
                f"Final dictionary mismatch for {qid}."
            )

        added = len(additions)
        total_added += added

        print(
            f"{qid.upper()}: VERIFIED | "
            f"Existing: {len(EXPECTED_EXISTING[qid])} | "
            f"Added: {added} | "
            f"Total: {len(final_relevance)}"
        )

    print("\nBATCH UPDATE COMPLETED")
    print(f"Backup: {backup_path}")
    print(f"Total new judgments: {total_added}")
    print("All five questions passed verification.")


if __name__ == "__main__":
    main()
