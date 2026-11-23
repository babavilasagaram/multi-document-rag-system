
import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from app.generation.generator import (
    generate_answer,
    prompt as original_prompt
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INPUT_PATH = Path(
    "evaluation/full_prompt_comparison_results.json"
)

OUTPUT_PATH = Path(
    "evaluation/completeness_rule_experiment_results.json"
)

EXPECTED_QUESTIONS = 16
EXPECTED_CHUNKS = 6

COMPLETENESS_RULE = """
10. When answering, include all distinct facts,
    recommendations, or steps in the provided context
    that directly address the question. Do not omit
    relevant supporting details. Combine overlapping
    information without repeating it. For questions
    asking for a single fact, answer that fact directly
    without adding unrelated information.
"""


# --------------------------------------------------
# 2. Validate input and output paths
# --------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(INPUT_PATH)

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing results: {OUTPUT_PATH}"
    )

data = json.loads(
    INPUT_PATH.read_text(encoding="utf-8")
)

records = data["results"]

if len(records) != EXPECTED_QUESTIONS:
    raise ValueError(
        f"Expected {EXPECTED_QUESTIONS} questions, "
        f"found {len(records)}."
    )

for record in records:
    if not record.get("frozen_chunks"):
        raise ValueError(
            f"{record['id']}: Missing frozen context."
        )

    if len(record["frozen_chunks"]) != EXPECTED_CHUNKS:
        raise ValueError(
            f"{record['id']}: Invalid chunk count."
        )

    if not record.get("original_prompt_answer"):
        raise ValueError(
            f"{record['id']}: Missing original answer."
        )


# --------------------------------------------------
# 3. Construct candidate prompt
# --------------------------------------------------

original_template = (
    original_prompt.messages[0].prompt.template
)

if original_template.count("CONTEXT:") != 1:
    raise ValueError(
        "Unexpected original prompt structure."
    )

candidate_template = original_template.replace(
    "CONTEXT:",
    COMPLETENESS_RULE + "\nCONTEXT:",
    1
)

candidate_prompt = ChatPromptTemplate.from_template(
    candidate_template
)

print("Original prompt imported from production generator.")
print("One completeness rule added.")
print("Production generator remains unchanged.")


# --------------------------------------------------
# 4. Generate candidate answers
# --------------------------------------------------

results = []

for index, record in enumerate(records, start=1):

    question_id = record["id"]
    question = record["question"]
    frozen_chunks = record["frozen_chunks"]

    print("\n" + "=" * 75)
    print(f"QUESTION {index}/{EXPECTED_QUESTIONS}: {question_id}")
    print(question)

    # Existing baseline answer
    original_answer = record["original_prompt_answer"]

    # Candidate answer using identical retrieval context
    candidate_answer = generate_answer(
        question,
        frozen_chunks,
        prompt_template=candidate_prompt
    )

    print("\nCANDIDATE ANSWER:")
    print(candidate_answer)

    results.append({
        "id": question_id,
        "question": question,
        "frozen_chunks": frozen_chunks,
        "original_prompt_answer": original_answer,
        "candidate_prompt_answer": candidate_answer
    })


# --------------------------------------------------
# 5. Save experiment results
# --------------------------------------------------

output_data = {
    "experiment": "Single completeness rule comparison",
    "dataset_name": data["dataset_name"],
    "model": "openai/gpt-oss-120b",
    "temperature": 0,
    "top_k": data["top_k"],
    "question_count": EXPECTED_QUESTIONS,
    "controls": {
        "same_questions": True,
        "same_frozen_contexts": True,
        "same_original_answers": True,
        "only_prompt_instruction_changed": True,
        "production_generator_modified": False
    },
    "added_rule": COMPLETENESS_RULE.strip(),
    "results": results
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT_PATH.open(
    "x",
    encoding="utf-8"
) as file:
    json.dump(
        output_data,
        file,
        indent=4,
        ensure_ascii=False
    )

# Post-save verification
saved_data = json.loads(
    OUTPUT_PATH.read_text(encoding="utf-8")
)

assert len(saved_data["results"]) == EXPECTED_QUESTIONS
assert all(
    item.get("candidate_prompt_answer")
    for item in saved_data["results"]
)

print("\n" + "=" * 75)
print("COMPLETENESS RULE EXPERIMENT COMPLETED")
print("Questions:", len(saved_data["results"]))
print("Results:", OUTPUT_PATH)
print("Post-save verification: PASSED")
print("=" * 75)
