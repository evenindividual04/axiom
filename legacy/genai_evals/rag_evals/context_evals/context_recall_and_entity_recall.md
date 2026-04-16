# Context Recall

## Overview

Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved. It focuses on not missing important results. Higher recall means fewer relevant documents were left out. In short, recall is about not missing anything important. Since it is about not missing anything, calculating context recall always requires a reference to compare against.

## LLM Based Context Recall

`LLMContextRecall` is computed using `user_input`, `reference` and the `retrieved_contexts`, and the values range between 0 and 1, with higher values indicating better performance. This metric uses `reference` as a proxy to `reference_contexts` which also makes it easier to use as annotating reference contexts can be very time-consuming.

To estimate context recall from the `reference`, the reference is broken down into claims each claim in the `reference` answer is analyzed to determine whether it can be attributed to the retrieved context or not. In an ideal scenario, all claims in the reference answer should be attributable to the retrieved context.

## Formula

The formula for calculating context recall is as follows:
```
Context Recall = (Number of claims in the reference supported by the retrieved context) / (Total number of claims in the reference)
```

## Key Points

- Values range from 0 to 1
- Higher values indicate better performance
- Measures completeness of retrieval
- Uses LLM to analyze claim attribution
- Easier to implement than manual annotation of reference contexts

# Context Entities Recall

## Overview

`ContextEntityRecall` metric gives the measure of recall of the retrieved context, based on the number of entities present in both `reference` and `retrieved_contexts` relative to the number of entities present in the `reference` alone. Simply put, it is a measure of what fraction of entities is recalled from `reference`.

## Use Cases

This metric is useful in fact-based use cases like:
- Tourism help desk
- Historical QA
- And more

This metric can help evaluate the retrieval mechanism for entities, based on comparison with entities present in `reference`, because in cases where entities matter, we need the `retrieved_contexts` which cover them.

## Calculation

To compute this metric, we use two sets:

- **RE**: The set of entities in the reference.
- **RCE**: The set of entities in the retrieved contexts.

We calculate the number of entities common to both sets (*RCE* ∩ *RE*) and divide it by the total number of entities in the reference (*RE*).

## Formula
```
Context Entity Recall = (Number of common entities between RCE and RE) / (Total number of entities in RE)
```

Where:
- *RCE* = Retrieved Context Entities
- *RE* = Reference Entities
- *RCE* ∩ *RE* = Intersection of entities in both sets

## Key Points

- Focuses on entity-level recall
- Values range from 0 to 1
- Higher values indicate better entity coverage
- Particularly useful for fact-based and knowledge-intensive applications