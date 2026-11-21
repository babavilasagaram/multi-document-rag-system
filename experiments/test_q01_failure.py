
import json
from pathlib import Path

from app.retrieval.retriever import (
    retrieve_documents,
    retrieve_multi_query
)

# Load expanded evaluation dataset
path = Path("data/evaluation_dataset_expansion_v1.json")

with open(path, "r", encoding="utf-8") as file:
    dataset = json.load(file)

item = next(
    question for question in dataset["queries"]
    if question["qid"] == "q01"
)

original_query = item["query"]

queries = [
    original_query,
    *item["query_variants"]
]

relevance = item["relevance"]

print("\nQUESTION:", original_query)
print("RELEVANCE:", relevance)


# Inspect individual query rankings
for index, query in enumerate(queries, start=1):

    print("\n" + "=" * 75)
    print(f"QUERY {index}: {query}")

    results = retrieve_documents(query, top_k=5)

    for rank, result in enumerate(results, start=1):

        chunk_id = result["id"]
        grade = relevance.get(str(chunk_id), 0)

        print(
            f"Rank: {rank} | "
            f"ID: {chunk_id} | "
            f"Similarity: {result['similarity']:.4f} | "
            f"Relevance: {grade}"
        )


# Inspect RRF ranking
print("\n" + "=" * 75)
print("RECIPROCAL RANK FUSION")

rrf_results = retrieve_multi_query(
    queries,
    top_k_per_query=5,
    final_k=10
)

for rank, result in enumerate(rrf_results, start=1):

    chunk_id = result["id"]
    grade = relevance.get(str(chunk_id), 0)

    print(
        f"Rank: {rank} | "
        f"ID: {chunk_id} | "
        f"RRF Score: {result['rrf_score']:.6f} | "
        f"Relevance: {grade}"
    )
