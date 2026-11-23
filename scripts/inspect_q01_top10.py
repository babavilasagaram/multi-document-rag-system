import json
import shutil
from pathlib import Path
from datetime import datetime

path = Path("data/evaluation_dataset_expansion_v1.json")

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

q01 = next(q for q in data["queries"] if q["qid"] == "q01")
grades = q01["relevance"]

expected = {
    "61": 2, "64": 2, "60": 1, "83": 2,
    "66": 1, "74": 0, "100": 0
}

if grades != expected:
    raise ValueError(f"Unexpected Q01 labels. No changes made: {grades}")

additions = {
    "62": 1,
    "65": 0,
    "76": 0,
    "81": 1,
    "89": 0,
    "90": 0,
    "93": 0,
    "101": 0
}

if any(chunk_id in grades for chunk_id in additions):
    raise ValueError("Proposed IDs already exist. No changes made.")

backup = path.with_name(
    f"{path.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
)
shutil.copy2(path, backup)

grades.update(additions)

with path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("Q01 updated successfully.")
print("Backup:", backup)
print("Final judgments:", dict(sorted(grades.items(), key=lambda x: int(x[0]))))