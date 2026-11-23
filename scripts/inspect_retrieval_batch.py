
import json
from pathlib import Path
from collections import defaultdict

from app.retrieval.retriever import (
    retrieve_documents,
    retrieve_multi_query,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation_dataset_expansion_v1.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_candidate_review_q15_q20.json"
)

BATCH_QIDS = ["q15", "q16", "q17", "q18", "q19", "q20"]
K = 10
TOP_K_PER_QUERY = 10



def get_chunk_id(chunk):
    return str(chunk["id"])


def get_chunk_content(chunk):
    return (
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("page_content")
        or "CONTENT FIELD NOT FOUND"
    )


def get_chunk_metadata(chunk):
    return chunk.get("metadata", {})


def get_chunk_grade(chunk_id, relevance):
    return relevance.get(chunk_id, "UNJUDGED")


def run_baseline(query):
    return retrieve_documents(query, top_k=K)


def run_candidate_pooling(queries):
    candidates = {}

    for query in queries:
        chunks = retrieve_documents(
            query,
            top_k=TOP_K_PER_QUERY,
        )

        for chunk in chunks:
            chunk_id = get_chunk_id(chunk)

            if (
                chunk_id not in candidates
                or chunk["similarity"]
                > candidates[chunk_id]["similarity"]
            ):
                candidates[chunk_id] = chunk

    ranked = sorted(
        candidates.values(),
        key=lambda chunk: (
            -chunk["similarity"],
            int(get_chunk_id(chunk)),
        ),
    )

    return ranked[:K]


def run_rrf(queries):
    return retrieve_multi_query(
        queries,
        top_k_per_query=TOP_K_PER_QUERY,
        final_k=K,
    )


def inspect_question(question):
    qid = question["qid"]
    query = question["query"]

    queries = [query] + question.get("query_variants", [])

    print("\n" + "=" * 75)
    print(f"{qid.upper()} | {query}")
    print("=" * 75)

    baseline = run_baseline(query)
    pooling = run_candidate_pooling(queries)
    rrf = run_rrf(queries)

    strategies = {
        "baseline": baseline,
        "candidate_pooling": pooling,
        "rrf": rrf,
    }

    union = {}

    for strategy_name, chunks in strategies.items():
        ids = [get_chunk_id(chunk) for chunk in chunks]

        print(f"\n{strategy_name.upper()}: {ids}")

        for chunk in chunks:
            chunk_id = get_chunk_id(chunk)

            if chunk_id not in union:
                union[chunk_id] = {
                    "chunk": chunk,
                    "retrieved_by": [],
                }

            union[chunk_id]["retrieved_by"].append(strategy_name)

    relevance = question.get("relevance", {})

    candidates = []

    for chunk_id in sorted(union, key=int):
        chunk = union[chunk_id]["chunk"]
        grade = get_chunk_grade(chunk_id, relevance)

        metadata = get_chunk_metadata(chunk)

        candidate = {
            "chunk_id": chunk_id,
            "grade": grade,
            "page": metadata.get("page", "Unknown"),
            "retrieved_by": union[chunk_id]["retrieved_by"],
            "content": get_chunk_content(chunk),
        }

        candidates.append(candidate)

    judged = [
        item for item in candidates
        if item["grade"] != "UNJUDGED"
    ]

    unjudged = [
        item for item in candidates
        if item["grade"] == "UNJUDGED"
    ]

    print(f"\nUNIQUE CANDIDATES: {len(candidates)}")
    print(f"ALREADY JUDGED: {len(judged)}")
    print(f"UNJUDGED: {len(unjudged)}")

    if not unjudged:
        print("\nNo unjudged candidates.")
    else:
        print("\nUNJUDGED CANDIDATE CONTENT:")

        for item in unjudged:
            print(f"\n--- CHUNK {item['chunk_id']} ---")
            print("Page:", item["page"])
            print("Retrieved by:", item["retrieved_by"])
            print(item["content"])

    return {
        "qid": qid,
        "query": query,
        "strategies": {
            name: [get_chunk_id(chunk) for chunk in chunks]
            for name, chunks in strategies.items()
        },
        "total_candidates": len(candidates),
        "already_judged_count": len(judged),
        "unjudged_count": len(unjudged),
        "candidates": candidates,
    }


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = {
        item["qid"]: item
        for item in dataset["queries"]
    }

    missing = [
        qid for qid in BATCH_QIDS
        if qid not in questions
    ]

    if missing:
        raise ValueError(f"Missing questions: {missing}")

    results = []

    for qid in BATCH_QIDS:
        result = inspect_question(questions[qid])
        results.append(result)

    report = {
        "experiment": "Retrieval candidate batch inspection",
        "batch": BATCH_QIDS,
        "top_k": K,
        "top_k_per_query": TOP_K_PER_QUERY,
        "dataset_modified": False,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 75)
    print("BATCH INSPECTION COMPLETED")
    print("=" * 75)
    print("Questions:", len(results))
    print(
        "Total unjudged candidates:",
        sum(item["unjudged_count"] for item in results),
    )
    print("Report saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
