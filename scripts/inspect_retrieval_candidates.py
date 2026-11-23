
import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.retriever import (
    retrieve_documents,
    retrieve_multi_query
)

DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation_dataset_expansion_v1.json"
)

K = 10
TOP_K_PER_QUERY = 10


def baseline_retrieval(query):
    return retrieve_documents(query, top_k=K)


def candidate_pooling_retrieval(queries):
    candidates = {}

    for query in queries:
        results = retrieve_documents(
            query,
            top_k=TOP_K_PER_QUERY
        )

        for result in results:
            chunk_id = result["id"]

            if chunk_id not in candidates:
                candidates[chunk_id] = result.copy()
            else:
                candidates[chunk_id]["similarity"] = max(
                    candidates[chunk_id]["similarity"],
                    result["similarity"]
                )

    ranked = sorted(
        candidates.values(),
        key=lambda x: (-x["similarity"], x["id"])
    )

    return ranked[:K]


def rrf_retrieval(queries):
    return retrieve_multi_query(
        queries,
        top_k_per_query=TOP_K_PER_QUERY,
        final_k=K
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "qid",
        help="Question ID, e.g. q01 or q02"
    )
    args = parser.parse_args()

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    item = next(
        (q for q in dataset["queries"] if q["qid"] == args.qid.lower()),
        None
    )

    if item is None:
        raise ValueError(f"Question ID not found: {args.qid}")

    original_query = item["query"]
    expanded_queries = [
        original_query,
        *item["query_variants"]
    ]

    grades = item["relevance"]

    strategies = {
        "BASELINE": baseline_retrieval(original_query),
        "CANDIDATE POOLING": candidate_pooling_retrieval(
            expanded_queries
        ),
        "RRF": rrf_retrieval(expanded_queries)
    }

    print("=" * 65)
    print(f"{item['qid'].upper()} | {original_query}")
    print("=" * 65)

    union = {}

    for name, results in strategies.items():
        ids = [str(chunk["id"]) for chunk in results]

        print(f"\n{name}: {ids}")

        for chunk in results:
            chunk_id = str(chunk["id"])
            union[chunk_id] = chunk

    print("\nUNJUDGED CANDIDATE CONTENT:")

    for chunk_id in sorted(union, key=int):
        if chunk_id not in grades:
            chunk = union[chunk_id]
            content = (
                chunk.get("content")
                or chunk.get("text")
                or chunk.get("page_content")
                or "CONTENT FIELD NOT FOUND"
            )
            metadata = chunk.get("metadata", {})

            print(f"\n--- CHUNK {chunk_id} ---")
            print("Page:", metadata.get("page", "Unknown"))
            print(content)

if __name__ == "__main__":
    main()
