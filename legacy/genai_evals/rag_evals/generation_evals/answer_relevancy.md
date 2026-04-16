# Answer Relevancy

The **Answer Relevancy** metric measures how relevant a response is to the user input. It ranges from 0 to 1, with higher scores indicating better alignment with the user input.

An answer is considered relevant if it directly and appropriately addresses the original question. This metric focuses on how well the answer matches the intent of the question, without evaluating factual accuracy. It penalizes answers that are incomplete or include unnecessary details.

This metric is calculated using the `user_input` and the `response` as follows:

1. Generate a set of artificial questions (default is 3) based on the response. These questions are designed to reflect the content of the response.

2. Compute the cosine similarity between the embedding of the user input ($E_o$) and the embedding of each generated question ($E_{g_i}$).

3. Take the average of these cosine similarity scores to get the **Answer Relevancy**:

$$
\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \text{cosine similarity}(E_{g_i}, E_o)
$$

$$
\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \frac{E_{g_i} \cdot E_o}{\|E_{g_i}\| \|E_o\|}
$$

## Where:

- $E_{g_i}$: Embedding of the $i^{th}$ generated question.
- $E_o$: Embedding of the user input.
- $N$: Number of generated questions (default is 3, configurable via `strictness` parameter).

## Note

While the score usually falls between 0 and 1, it is not guaranteed due to cosine similarity's mathematical range of -1 to 1.