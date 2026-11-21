
from app.retrieval.retriever import (
    retrieve_documents,
    retrieve_multi_query
)


# ==================================================
# 1. EVALUATION DATASET
# ==================================================

evaluation_data = [
    {
        "query": "Why does the Transformer not use recurrence?",
        "relevant_ids": [61, 64],
        "variations": [
            "Why does the Transformer not use recurrence?",
            "How does the Transformer process sequences without recurrent layers?",
            "Why can Transformer process tokens in parallel?"
        ]
    },
    {
        "query": "How does self-attention allow parallel computation?",
        "relevant_ids": [60, 81],
        "variations": [
            "How does self-attention allow parallel computation?",
            "Why can self-attention process all positions simultaneously?",
            "How does the Transformer avoid sequential computation?"
        ]
    },
    {
        "query": "What are advantages of multi-head attention compared to single-head attention?",
        "relevant_ids": [73],
        "variations": [
            "What are advantages of multi-head attention compared to single-head attention?",
            "Why does the Transformer use multiple attention heads?",
            "How do multiple attention heads learn different representations?"
        ]
    },
    {
        "query": "How does scaled dot-product attention work?",
        "relevant_ids": [69, 70, 71],
        "variations": [
            "How does scaled dot-product attention work?",
            "How are queries keys and values used in attention?",
            "Why are attention scores divided by the square root of key dimension?"
        ]
    },
    {
        "query": "Why does Transformer need positional encoding?",
        "relevant_ids": [79],
        "variations": [
            "Why does Transformer need positional encoding?",
            "How does Transformer represent the order of tokens?",
            "How does Transformer capture sequence order without recurrence?"
        ]
    }
]


K = 5
TOP_K_PER_QUERY = 5


# ==================================================
# 2. CANDIDATE POOLING
# ==================================================

def retrieve_candidate_pool(queries, top_k_per_query=5, final_k=5):

    if not queries:
        raise ValueError("At least one query is required.")

    unique_queries = list(dict.fromkeys(
        query.strip()
        for query in queries
        if query and query.strip()
    ))

    if not unique_queries:
        raise ValueError("No valid queries provided.")

    candidates = {}

    for query in unique_queries:

        results = retrieve_documents(
            query,
            top_k=top_k_per_query
        )

        for chunk in results:

            chunk_id = chunk["id"]

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    **chunk,
                    "max_similarity": chunk["similarity"]
                }

            else:

                candidates[chunk_id]["max_similarity"] = max(
                    candidates[chunk_id]["max_similarity"],
                    chunk["similarity"]
                )

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda x: (
            -x["max_similarity"],
            x["id"]
        )
    )

    return ranked_candidates[:final_k]


# ==================================================
# 3. METRIC CALCULATION
# ==================================================

def calculate_metrics(results, relevant_ids, k=5):

    retrieved_ids = [
        result["id"]
        for result in results[:k]
    ]

    relevant = set(relevant_ids)
    retrieved = set(retrieved_ids)

    hits = retrieved.intersection(relevant)

    recall = (
        len(hits) / len(relevant)
        if relevant else 0
    )

    precision = len(hits) / k

    reciprocal_rank = 0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):

        if chunk_id in relevant:
            reciprocal_rank = 1 / rank
            break

    hit_rate = int(bool(hits))

    return {
        "recall": recall,
        "precision": precision,
        "mrr": reciprocal_rank,
        "hit_rate": hit_rate
    }


# ==================================================
# 4. THREE RETRIEVAL STRATEGIES
# ==================================================

strategies = {
    "Single Query": lambda item: retrieve_documents(
        item["query"],
        top_k=K
    ),

    "Candidate Pooling": lambda item: retrieve_candidate_pool(
        item["variations"],
        top_k_per_query=TOP_K_PER_QUERY,
        final_k=K
    ),

    "Multi Query RRF": lambda item: retrieve_multi_query(
        item["variations"],
        top_k_per_query=TOP_K_PER_QUERY,
        final_k=K
    )
}


all_metrics = {
    strategy: []
    for strategy in strategies
}


# ==================================================
# 5. RUN COMPARISON
# ==================================================

for query_index, item in enumerate(evaluation_data, start=1):

    print("\n\n" + "=" * 85)
    print(f"QUERY {query_index}: {item['query']}")
    print(f"EXPECTED RELEVANT IDS: {item['relevant_ids']}")
    print("=" * 85)

    for strategy_name, retrieve in strategies.items():

        results = retrieve(item)

        metrics = calculate_metrics(
            results,
            item["relevant_ids"],
            K
        )

        all_metrics[strategy_name].append(metrics)

        print("\n" + "-" * 75)
        print(f"STRATEGY: {strategy_name}")
        print("-" * 75)

        print(
            "Retrieved IDs:",
            [result["id"] for result in results]
        )

        print(
            f"Recall@{K}: {metrics['recall']:.3f} | "
            f"Precision@{K}: {metrics['precision']:.3f} | "
            f"MRR: {metrics['mrr']:.3f} | "
            f"Hit Rate: {metrics['hit_rate']}"
        )

        print("\nRetrieved chunk details:")

        for rank, result in enumerate(results, start=1):

            print(
                f"Rank {rank} | "
                f"ID {result['id']} | "
                f"Page {result['metadata'].get('page')} | "
                f"Similarity {result['similarity']:.4f}",
                end=""
            )

            if "max_similarity" in result:
                print(
                    f" | Max Similarity: "
                    f"{result['max_similarity']:.4f}",
                    end=""
                )

            if "rrf_score" in result:
                print(
                    f" | RRF Score: "
                    f"{result['rrf_score']:.5f}",
                    end=""
                )

            print()


# ==================================================
# 6. AGGREGATE COMPARISON
# ==================================================

print("\n\n" + "=" * 85)
print("FINAL THREE-WAY COMPARISON")
print("=" * 85)

for strategy_name, metrics_list in all_metrics.items():

    print(f"\n{strategy_name}")
    print("-" * 50)

    for metric in [
        "recall",
        "precision",
        "mrr",
        "hit_rate"
    ]:

        average = sum(
            item[metric]
            for item in metrics_list
        ) / len(metrics_list)

        print(f"Mean {metric}: {average:.3f}")
