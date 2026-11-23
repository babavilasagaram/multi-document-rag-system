
import json
from pathlib import Path

from app.retrieval.retriever import retrieve_documents
from app.generation.generator import (
    generate_answer,
    prompt as original_prompt
)


# --------------------------------------------------
# 1. Load existing frozen G09 context
# --------------------------------------------------

input_path = Path(
    "evaluation/controlled_prompt_comparison_results.json"
)

output_path = Path(
    "evaluation/g09_context_addition_results.json"
)

with input_path.open("r", encoding="utf-8") as file:
    previous_results = json.load(file)

g09 = next(
    item for item in previous_results
    if item["id"] == "G09"
)

question = g09["question"]

baseline_chunks = g09["retrieved_chunks"]


# --------------------------------------------------
# 2. Retrieve additional candidate chunks
# --------------------------------------------------

retrieved = retrieve_documents(
    question,
    top_k=10
)

existing_ids = {
    str(chunk["id"])
    for chunk in baseline_chunks
}

additional_chunk = next(
    (
        chunk for chunk in retrieved
        if str(chunk["id"]) == "231"
    ),
    None
)

if additional_chunk is None:
    raise ValueError(
        "Chunk 231 was not returned by current retrieval."
    )

if str(additional_chunk["id"]) in existing_ids:
    raise ValueError(
        "Chunk 231 already exists in baseline context."
    )


# --------------------------------------------------
# 3. Construct two contexts
# --------------------------------------------------

expanded_chunks = baseline_chunks + [additional_chunk]

print("=" * 80)
print("G09 CONTEXT ADDITION EXPERIMENT")
print("=" * 80)

print("\nBaseline chunk IDs:")
print([chunk["id"] for chunk in baseline_chunks])

print("\nExpanded chunk IDs:")
print([chunk["id"] for chunk in expanded_chunks])


# --------------------------------------------------
# 4. Generate with baseline context
# --------------------------------------------------

print("\n" + "=" * 80)
print("EXPERIMENT A: ORIGINAL SIX CHUNKS")
print("=" * 80)

baseline_answer = generate_answer(
    question,
    baseline_chunks,
    prompt_template=original_prompt
)

print("\nBASELINE ANSWER:")
print(baseline_answer)


# --------------------------------------------------
# 5. Generate with additional chunk
# --------------------------------------------------

print("\n" + "=" * 80)
print("EXPERIMENT B: SIX CHUNKS + CHUNK 231")
print("=" * 80)

expanded_answer = generate_answer(
    question,
    expanded_chunks,
    prompt_template=original_prompt
)

print("\nEXPANDED CONTEXT ANSWER:")
print(expanded_answer)


# --------------------------------------------------
# 6. Save experiment
# --------------------------------------------------

results = {
    "id": "G09",
    "question": question,
    "experiment": "Additional context chunk",
    "baseline_chunk_ids": [
        chunk["id"] for chunk in baseline_chunks
    ],
    "expanded_chunk_ids": [
        chunk["id"] for chunk in expanded_chunks
    ],
    "baseline_answer": baseline_answer,
    "expanded_answer": expanded_answer,
    "controls": {
        "same_question": True,
        "same_prompt": True,
        "baseline_chunks_preserved": True,
        "one_additional_chunk": True
    }
}

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nResults saved to:", output_path)
