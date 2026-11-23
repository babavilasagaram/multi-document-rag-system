
import json
import shutil
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
dataset_path = project_root / "data" / "evaluation_dataset_expansion_v1.json"

approved_additions = {
    "65": 0,
    "66": 0,
    "74": 0,
    "90": 0,
    "93": 0,
    "94": 0,
}

with open(dataset_path, "r", encoding="utf-8") as file:
    dataset = json.load(file)

q12 = next(
    (item for item in dataset["queries"] if item["qid"] == "q12"),
    None
)

if q12 is None:
    raise ValueError("Q12 not found. No changes made.")

relevance = q12["relevance"]

# Never silently overwrite an existing positive judgment.
for chunk_id, new_grade in approved_additions.items():
    existing_grade = relevance.get(chunk_id)

    if existing_grade is not None and existing_grade != new_grade:
        raise ValueError(
            f"Conflict for Q12 chunk {chunk_id}: "
            f"existing grade={existing_grade}, proposed grade={new_grade}. "
            "No changes made."
        )

# Make a backup before modifying the dataset.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = dataset_path.with_name(
    f"{dataset_path.stem}.before_q12_audit_{timestamp}.json"
)
shutil.copy2(dataset_path, backup_path)

changes = []

for chunk_id, grade in approved_additions.items():
    if chunk_id not in relevance:
        relevance[chunk_id] = grade
        changes.append({
            "qid": "q12",
            "chunk_id": chunk_id,
            "old_grade": None,
            "new_grade": grade,
        })

batch_name = "Q12 retrieved candidate audit"

history = dataset.setdefault("review_history", [])

if not any(entry.get("batch") == batch_name for entry in history):
    history.append({
        "batch": batch_name,
        "decision": "User approved all six proposed Q12 candidate judgments",
        "changes": changes,
    })

with open(dataset_path, "w", encoding="utf-8") as file:
    json.dump(dataset, file, indent=2, ensure_ascii=False)
    file.write("\n")

# Verify the saved file.
with open(dataset_path, "r", encoding="utf-8") as file:
    verified = json.load(file)

verified_q12 = next(
    item for item in verified["queries"] if item["qid"] == "q12"
)

for chunk_id, grade in approved_additions.items():
    assert verified_q12["relevance"][chunk_id] == grade

print("Q12 candidate audit update successful.")
print("New judgments added:", len(changes))
print("Q12 relevance:", verified_q12["relevance"])
print("Backup:", backup_path)
