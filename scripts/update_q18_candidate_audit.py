import json
import shutil
from datetime import datetime
from pathlib import Path

dataset_path = Path("data/evaluation_dataset_expansion_v1.json")

additions = {
    "107": 0,
    "87": 0,
    "71": 0,
    "91": 0,
    "93": 0,
    "89": 0,
    "90": 0,
    "92": 0,
}

data = json.loads(dataset_path.read_text(encoding="utf-8"))

matches = [q for q in data["queries"] if q["qid"] == "q18"]
if len(matches) != 1:
    raise ValueError("Expected exactly one Q18 entry.")

q18 = matches[0]
relevance = q18["relevance"]

# Never overwrite an existing judgment.
for chunk_id, grade in additions.items():
    if chunk_id in relevance and relevance[chunk_id] != grade:
        raise ValueError(
            f"Conflict: chunk {chunk_id} already has grade "
            f"{relevance[chunk_id]}"
        )

missing = {
    chunk_id: grade
    for chunk_id, grade in additions.items()
    if chunk_id not in relevance
}

if not missing:
    print("Q18 audit already complete. No changes needed.")
    raise SystemExit(0)

backup = dataset_path.with_name(
    f"{dataset_path.stem}.before_q18_audit_"
    f"{datetime.now():%Y%m%d_%H%M%S}.json"
)

shutil.copy2(dataset_path, backup)

relevance.update(missing)

data.setdefault("review_history", []).append({
    "batch": "Q18 candidate audit",
    "decision": "User approved eight grade-0 candidate additions",
    "changes": [
        {
            "qid": "q18",
            "chunk_id": chunk_id,
            "old_grade": None,
            "new_grade": grade,
        }
        for chunk_id, grade in missing.items()
    ],
})

dataset_path.write_text(
    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

# Verify the saved file.
verified = json.loads(dataset_path.read_text(encoding="utf-8"))
saved = next(q for q in verified["queries"] if q["qid"] == "q18")

assert all(saved["relevance"].get(k) == v for k, v in additions.items())

print("Q18 candidate audit update successful.")
print("New judgments added:", len(missing))
print("Q18 relevance:", saved["relevance"])
print("Backup:", backup.resolve())