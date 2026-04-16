# Context Precision Metric

Context Precision is a metric that evaluates the retriever's ability to rank relevant chunks higher than irrelevant ones for a given query in the retrieved context. Specifically, it assesses the degree to which relevant chunks in the retrieved context are placed at the top of the ranking.

## Calculation

It is calculated as the mean of the precision@k for each chunk in the context. Precision@k is the ratio of the number of relevant chunks at rank k to the total number of chunks at rank k.

### Formula
```
Context Precision@K = (Σ(k=1 to K) (Precision@k × vₖ)) / (Total number of relevant items in the top K results)
```

Where:
```
Precision@k = (true positives@k) / (true positives@k + false positives@k)
```

### Parameters

- **K**: The total number of chunks in `retrieved_contexts`
- **vₖ ∈ {0, 1}**: The relevance indicator at rank k

### Description

This metric provides a weighted evaluation of how well the retrieval system ranks relevant information, with higher weights given to relevant items that appear earlier in the results.

## Context Utilization
The ContextUtilization metric evaluates whether retrieved contexts are useful by comparing each context against the generated response. Use this when you don't have a reference answer but have the response that was generated.
Note that even if an irrelevant chunk is present at the second position in the array, context precision remains the same. However, if this irrelevant chunk is placed at the first position, context precision reduces: