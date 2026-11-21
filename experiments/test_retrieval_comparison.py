
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


# ==================================================
# 2. METRIC CALCULATION
# ==================================================

def calculate_metrics(results, relevant_ids, k=5):

    retrieved_ids = [
        result["id"] for result in results[:k]
    ]

    relevant = set(relevant_ids)
    retrieved = set(retrieved_ids)

    hits = retrieved.intersection(relevant)

    recall = (
        len(hits) / len(relevant)
        if relevant else 0
    )

    precision = (
        len(hits) / k
    )

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
# 3. RETRIEVAL STRATEGIES
# ==================================================

strategies = {
    "Single Query": lambda item: retrieve_documents(
        item["query"],
        top_k=K
    ),

    "Multi Query RRF": lambda item: retrieve_multi_query(
        item["variations"],
        top_k_per_query=5,
        final_k=K
    )
}


all_results = {
    strategy: []
    for strategy in strategies
}


# ==================================================
# 4. PRINT RETRIEVED CHUNK DETAILS
# ==================================================

def display_results(results):

    for rank, result in enumerate(results, start=1):

        print("\n" + "-" * 75)

        print(f"Rank: {rank}")
        print(f"Chunk ID: {result['id']}")
        print(f"Page: {result['metadata'].get('page')}")
        print(f"Similarity: {result['similarity']:.4f}")

        if "rrf_score" in result:
            print(f"RRF Score: {result['rrf_score']:.5f}")

        print("\nChunk Text:")
        print(result["text"])

    print("\n" + "-" * 75)


# ==================================================
# 5. RUN EVALUATION
# ==================================================

for query_index, item in enumerate(evaluation_data, start=1):

    print("\n\n" + "=" * 90)
    print(f"QUERY {query_index}: {item['query']}")
    print(f"EXPECTED RELEVANT IDS: {item['relevant_ids']}")
    print("=" * 90)

    for strategy_name, retrieve in strategies.items():

        results = retrieve(item)

        metrics = calculate_metrics(
            results,
            item["relevant_ids"],
            K
        )

        all_results[strategy_name].append(metrics)

        print("\n\n" + "#" * 90)
        print(f"STRATEGY: {strategy_name}")
        print("#" * 90)

        print(
            "\nRetrieved IDs:",
            [result["id"] for result in results]
        )

        print("\nMETRICS:")
        print(f"Recall@{K}:    {metrics['recall']:.3f}")
        print(f"Precision@{K}: {metrics['precision']:.3f}")
        print(f"MRR:           {metrics['mrr']:.3f}")
        print(f"Hit Rate@{K}:  {metrics['hit_rate']}")

        print("\nRETRIEVED CHUNKS:")
        display_results(results)


# ==================================================
# 6. AGGREGATE COMPARISON
# ==================================================

print("\n\n" + "=" * 90)
print("FINAL AGGREGATE COMPARISON")
print("=" * 90)

for strategy_name, metrics_list in all_results.items():

    print(f"\n{strategy_name}")
    print("-" * 50)

    for metric in [
        "recall",
        "precision",
        "mrr",
        "hit_rate"
    ]:

        average = sum(
            item[metric] for item in metrics_list
        ) / len(metrics_list)

        print(f"Mean {metric}: {average:.3f}")
