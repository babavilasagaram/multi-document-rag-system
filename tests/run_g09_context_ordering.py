
import copy
import json
from pathlib import Path

from app.generation.generator import (
    generate_answer,
    prompt as original_prompt
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INPUT_PATH = Path(
    "evaluation/controlled_prompt_comparison_results.json"
)

OUTPUT_PATH = Path(
    "evaluation/g09_context_ordering_results.json"
)

TEST_ID = "G09"


# --------------------------------------------------
# 2. Load the frozen retrieval context
# --------------------------------------------------

with INPUT_PATH.open("r", encoding="utf-8") as file:
    previous_results = json.load(file)

test_data = next(
    (item for item in previous_results if item["id"] == TEST_ID),
    None
)

if test_data is None:
    raise ValueError(f"{TEST_ID} not found in input file.")

question = test_data["question"]

original_chunks = copy.deepcopy(
    test_data["retrieved_chunks"]
)

if not original_chunks:
    raise ValueError("No retrieved chunks found.")


# --------------------------------------------------
# 3. Create grouped ordering
# --------------------------------------------------

def get_source(chunk):
    metadata = chunk.get("metadata") or {}
    return str(metadata.get("source", "UNKNOWN"))


def get_page(chunk):
    metadata = chunk.get("metadata") or {}

    page = metadata.get("page")

    if page is None:
        page = metadata.get("page_number")

    try:
        return int(page)
    except (TypeError, ValueError):
        return 10**9


# Python's sort is stable. Chunks from the same source
# and page retain their relative order.
grouped_chunks = sorted(
    copy.deepcopy(original_chunks),
    key=lambda chunk: (
        get_source(chunk).lower(),
        get_page(chunk)
    )
)


# --------------------------------------------------
# 4. Verify that no chunks were added or removed
# --------------------------------------------------

def chunk_ids(chunks):
    return sorted(
        str(chunk.get("id"))
        for chunk in chunks
    )


if chunk_ids(original_chunks) != chunk_ids(grouped_chunks):
    raise RuntimeError(
        "Chunk IDs differ. Experiment is not controlled."
    )


print("=" * 80)
print("G09 CONTEXT ORDERING EXPERIMENT")
print("=" * 80)

print(f"\nQuestion: {question}")
print(f"Original chunks: {len(original_chunks)}")
print(f"Grouped chunks: {len(grouped_chunks)}")

print("\nOriginal chunk order:")

for position, chunk in enumerate(original_chunks, start=1):
    print(
        position,
        chunk.get("id"),
        get_source(chunk),
        get_page(chunk)
    )

print("\nGrouped chunk order:")

for position, chunk in enumerate(grouped_chunks, start=1):
    print(
        position,
        chunk.get("id"),
        get_source(chunk),
        get_page(chunk)
    )


# --------------------------------------------------
# 5. Run original prompt on original ordering
# --------------------------------------------------

print("\n" + "=" * 80)
print("EXPERIMENT A: ORIGINAL ORDER")
print("=" * 80)

original_answer = generate_answer(
    question,
    original_chunks,
    prompt_template=original_prompt
)

print("\nORIGINAL ORDER ANSWER:")
print(original_answer)


# --------------------------------------------------
# 6. Run original prompt on grouped ordering
# --------------------------------------------------

print("\n" + "=" * 80)
print("EXPERIMENT B: GROUPED BY SOURCE AND PAGE")
print("=" * 80)

grouped_answer = generate_answer(
    question,
    grouped_chunks,
    prompt_template=original_prompt
)

print("\nGROUPED ORDER ANSWER:")
print(grouped_answer)


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

results = {
    "id": TEST_ID,
    "question": question,
    "experiment": "Context ordering only",
    "prompt": "Original baseline prompt",
    "original_chunks": original_chunks,
    "grouped_chunks": grouped_chunks,
    "original_order_answer": original_answer,
    "grouped_order_answer": grouped_answer,
    "controls": {
        "same_question": True,
        "same_chunk_ids": True,
        "same_chunk_text": True,
        "same_prompt": True,
        "retrieval_repeated": False,
        "only_order_changed": True
    }
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETED")
print("=" * 80)

print(f"Results saved to: {OUTPUT_PATH}")
print("Generation calls: 2")
print("Retrieval calls: 0")
