
import json
import math
from pathlib import Path

from app.retrieval.retriever import (
    retrieve_documents,
    retrieve_multi_query
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_PATH = (
    PROJECT_ROOT / "data" / "evaluation_dataset_expansion_v1.json"
)

K = 5
TOP_K_PER_QUERY = 5


with open(EVALUATION_PATH, "r", encoding="utf-8") as file:
    evaluation_dataset = json.load(file)

evaluation_data = evaluation_dataset["queries"]

print("=" * 75)
print("RAG RETRIEVAL EVALUATION")
print("=" * 75)

print("Dataset:", evaluation_dataset["dataset_name"])
print("Status:", evaluation_dataset["status"])
print("Questions:", len(evaluation_data))
print("K:", K)


# ==================================================
# 2. EVALUATION METRICS
# ==================================================

def calculate_ndcg(retrieved_ids, relevance, k=5):

    retrieved_ids = retrieved_ids[:k]

    def dcg(scores):
        return sum(
            (2 ** score - 1) / math.log2(rank + 1)
            for rank, score in enumerate(scores, start=1)
        )

    actual_scores = [
        int(relevance.get(str(chunk_id), 0))
        for chunk_id in retrieved_ids
    ]

    ideal_scores = sorted(
        [int(grade) for grade in relevance.values()],
        reverse=True
    )[:k]

    actual_dcg = dcg(actual_scores)
    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def calculate_metrics(retrieved_ids, relevance, k=5):

    retrieved_ids = retrieved_ids[:k]

    relevant_set = {
        int(chunk_id)
        for chunk_id, grade in relevance.items()
        if int(grade) > 0
    }

    retrieved_set = set(retrieved_ids)

    relevant_retrieved = retrieved_set.intersection(
        relevant_set
    )

    recall = (
        len(relevant_retrieved) / len(relevant_set)
        if relevant_set else 0.0
    )

    precision = len(relevant_retrieved) / k

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            reciprocal_rank = 1 / rank
            break

    hit = 1 if relevant_retrieved else 0

    ndcg = calculate_ndcg(
        retrieved_ids,
        relevance,
        k
    )

    return {
        "recall": recall,
        "precision": precision,
        "mrr": reciprocal_rank,
        "hit": hit,
        "ndcg": ndcg
    }


# ==================================================
# 3. RETRIEVAL STRATEGIES
# ==================================================

def baseline_retrieval(query):

    return retrieve_documents(
        query,
        top_k=K
    )


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

    ranked_results = sorted(
        candidates.values(),
        key=lambda x: (
            -x["similarity"],
            x["id"]
        )
    )

    return ranked_results[:K]


def rrf_retrieval(queries):

    return retrieve_multi_query(
        queries,
        top_k_per_query=TOP_K_PER_QUERY,
        final_k=K
    )


# ==================================================
# 4. EVALUATE ALL STRATEGIES
# ==================================================

strategies = {
    "Baseline": baseline_retrieval,
    "Candidate Pooling": candidate_pooling_retrieval,
    "RRF": rrf_retrieval
}

all_results = {
    strategy: []
    for strategy in strategies
}


for item in evaluation_data:

    qid = item["qid"]
    original_query = item["query"]
    query_variants = item["query_variants"]
    relevance = item["relevance"]

    # Same expanded query set for both fusion strategies
    expanded_queries = [
        original_query,
        *query_variants
    ]

    print("\n" + "=" * 75)
    print(f"{qid.upper()} | {original_query}")

    print("Expanded queries:", len(expanded_queries))

    for strategy_name, retrieval_function in strategies.items():

        if strategy_name == "Baseline":

            results = retrieval_function(original_query)

        else:

            results = retrieval_function(expanded_queries)

        retrieved_ids = [
            result["id"]
            for result in results
        ]

        metrics = calculate_metrics(
            retrieved_ids,
            relevance,
            K
        )

        all_results[strategy_name].append({
            "qid": qid,
            "retrieved_ids": retrieved_ids,
            "metrics": metrics
        })

        print(f"\n{strategy_name}")
        print("Retrieved:", retrieved_ids)

        print(
            f"Recall: {metrics['recall']:.3f} | "
            f"Precision: {metrics['precision']:.3f} | "
            f"MRR: {metrics['mrr']:.3f} | "
            f"Hit: {metrics['hit']} | "
            f"nDCG: {metrics['ndcg']:.3f}"
        )


# ==================================================
# 5. AGGREGATE RESULTS
# ==================================================

metric_names = [
    "recall",
    "precision",
    "mrr",
    "hit",
    "ndcg"
]

average_results = {}

for strategy_name, question_results in all_results.items():

    num_queries = len(question_results)

    averages = {
        metric: sum(
            result["metrics"][metric]
            for result in question_results
        ) / num_queries
        for metric in metric_names
    }

    average_results[strategy_name] = averages


# ==================================================
# 6. FINAL COMPARISON
# ==================================================

print("\n\n" + "=" * 90)
print("FINAL RETRIEVAL STRATEGY COMPARISON")
print("=" * 90)

print(
    f"{'Strategy':<22}"
    f"{'Recall@5':>12}"
    f"{'Precision@5':>14}"
    f"{'MRR':>10}"
    f"{'Hit Rate':>12}"
    f"{'nDCG@5':>12}"
)

print("-" * 90)

for strategy_name, metrics in average_results.items():

    print(
        f"{strategy_name:<22}"
        f"{metrics['recall']:>12.3f}"
        f"{metrics['precision']:>14.3f}"
        f"{metrics['mrr']:>10.3f}"
        f"{metrics['hit']:>12.3f}"
        f"{metrics['ndcg']:>12.3f}"
    )

print("=" * 90)


# ==================================================
# 7. PER-QUESTION COMPARISON
# ==================================================

print("\nPER-QUESTION nDCG COMPARISON")
print("-" * 65)

print(
    f"{'QID':<10}"
    f"{'Baseline':>15}"
    f"{'Pooling':>15}"
    f"{'RRF':>15}"
)

for index, item in enumerate(evaluation_data):

    baseline = all_results["Baseline"][index]["metrics"]["ndcg"]
    pooling = all_results["Candidate Pooling"][index]["metrics"]["ndcg"]
    rrf = all_results["RRF"][index]["metrics"]["ndcg"]

    print(
        f"{item['qid']:<10}"
        f"{baseline:>15.3f}"
        f"{pooling:>15.3f}"
        f"{rrf:>15.3f}"
    )


# ==================================================
# 8. SAVE RESULTS
# ==================================================

output_path = PROJECT_ROOT / "data" / "retrieval_comparison_results.json"

output_data = {
    "dataset": evaluation_dataset["dataset_name"],
    "dataset_status": evaluation_dataset["status"],
    "k": K,
    "top_k_per_query": TOP_K_PER_QUERY,
    "strategies": {
        strategy: {
            "average_metrics": average_results[strategy],
            "per_question": all_results[strategy]
        }
        for strategy in strategies
    }
}

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=4)

print("\nResults saved to:", output_path)
