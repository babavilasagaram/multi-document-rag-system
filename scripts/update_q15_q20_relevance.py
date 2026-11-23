
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_dataset_expansion_v1.json"

EXPECTED_EXISTING = {
    "q15": {
        "61": 1, "82": 2, "83": 2, "84": 1, "85": 1,
        "78": 2, "81": 1, "63": 1, "74": 0, "80": 0, "86": 0
    },
    "q16": {
        "78": 2, "83": 2, "84": 1, "85": 1, "81": 1, "80": 0
    },
    "q17": {
        "86": 2, "87": 2, "55": 0, "98": 0, "91": 0,
        "92": 0, "112": 0, "103": 0, "113": 0
    },
    "q18": {
        "88": 2, "107": 0, "87": 0, "71": 0, "91": 0,
        "93": 0, "89": 0, "90": 0, "92": 0
    },
    "q19": {
        "89": 2, "91": 2, "90": 2, "55": 2, "56": 1, "93": 0
    },
    "q20": {
        "80": 2, "95": 2, "96": 2, "79": 1,
        "76": 0, "77": 0, "75": 0, "68": 0
    },
}

APPROVED_ADDITIONS = {
    "q15": {
        "60": 1,
        "68": 0,
        "102": 0
    },
    "q16": {
        "60": 1,
        "62": 1,
        "63": 0,
        "70": 0,
        "74": 0,
        "82": 1,
        "86": 0,
        "106": 0
    },
    "q17": {
        "90": 0,
        "101": 1,
        "109": 0
    },
    "q18": {
        "58": 0,
        "66": 0,
        "73": 0,
        "74": 0,
        "100": 0,
        "377": 0
    },
    "q19": {
        "92": 0,
        "94": 0,
        "98": 0,
        "101": 1
    },
    "q20": {
        "57": 0,
        "65": 0,
        "81": 0,
        "89": 0,
        "94": 1,
        "112": 0
    },
}


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    queries = {
        item["qid"]: item
        for item in dataset["queries"]
    }

    # Verify all target questions exist.
    for qid in EXPECTED_EXISTING:
        if qid not in queries:
            raise ValueError(f"{qid} is missing from the dataset.")

    # Verify exact existing judgments.
    for qid, expected in EXPECTED_EXISTING.items():
        actual = queries[qid]["relevance"]

        if actual != expected:
            raise ValueError(
                f"\nSAFETY CHECK FAILED for {qid}.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}\n"
                "Dataset was not modified."
            )

    # Prevent overwriting existing judgments.
    for qid, additions in APPROVED_ADDITIONS.items():
        existing_ids = set(queries[qid]["relevance"])
        overlap = existing_ids.intersection(additions)

        if overlap:
            raise ValueError(
                f"{qid}: existing IDs would be overwritten: "
                f"{sorted(overlap)}"
            )

        for chunk_id, grade in additions.items():
            if grade not in (0, 1, 2):
                raise ValueError(
                    f"Invalid grade {grade}: {qid}, chunk {chunk_id}"
                )

    # Create backup.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DATASET_PATH.with_name(
        f"evaluation_dataset_expansion_v1_backup_{timestamp}.json"
    )

    if backup_path.exists():
        raise FileExistsError(backup_path)

    shutil.copy2(DATASET_PATH, backup_path)

    # Apply approved additions.
    for qid, additions in APPROVED_ADDITIONS.items():
        queries[qid]["relevance"].update(additions)

    # Save.
    with DATASET_PATH.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, ensure_ascii=False)
        file.write("\n")

    # Reopen and verify saved dataset.
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        saved_dataset = json.load(file)

    saved_queries = {
        item["qid"]: item
        for item in saved_dataset["queries"]
    }

    total_added = 0

    for qid, additions in APPROVED_ADDITIONS.items():
        final_relevance = saved_queries[qid]["relevance"]
        expected_final = EXPECTED_EXISTING[qid] | additions

        if final_relevance != expected_final:
            raise RuntimeError(
                f"Final dictionary verification failed for {qid}."
            )

        for chunk_id, grade in additions.items():
            if final_relevance.get(chunk_id) != grade:
                raise RuntimeError(
                    f"Grade verification failed: {qid}, {chunk_id}"
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
    print("All six questions passed verification.")


if __name__ == "__main__":
    main()
