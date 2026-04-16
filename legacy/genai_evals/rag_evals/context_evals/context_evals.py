import asyncio
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    ContextUtilization
)
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class RAGContextEvaluator:
    """
    A unified class to evaluate RAG systems using multiple RAGAS metrics.

    Metrics included:
    - Context Entity Recall: Measures how many entities from reference are in retrieved contexts
    - Context Precision: Measures relevance of retrieved contexts to the query
    - Context Recall: Measures if retrieved contexts contain information to answer the query
    - Context Utilization: Measures how well the response uses the retrieved contexts
    """

    def __init__(self, model_name: str = "gpt-4o-mini", client: Optional[AsyncOpenAI] = None):
        self.client = client or AsyncOpenAI()
        self.llm = llm_factory(model_name, client=self.client)

        # Initialize all metrics
        self.context_entity_recall = ContextEntityRecall(llm=self.llm)
        self.context_precision = ContextPrecision(llm=self.llm)
        self.context_recall = ContextRecall(llm=self.llm)
        self.context_utilization = ContextUtilization(llm=self.llm)

    async def evaluate_context_entity_recall(
            self,
            reference: str,
            retrieved_contexts: List[str]
    ) -> float:
        result = await self.context_entity_recall.ascore(
            reference=reference,
            retrieved_contexts=retrieved_contexts
        )
        return result.value

    async def evaluate_context_precision(
            self,
            user_input: str,
            reference: str,
            retrieved_contexts: List[str]
    ) -> float:
        result = await self.context_precision.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts
        )
        return result.value

    async def evaluate_context_recall(
            self,
            user_input: str,
            reference: str,
            retrieved_contexts: List[str]
    ) -> float:
        result = await self.context_recall.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts
        )
        return result.value

    async def evaluate_context_utilization(
            self,
            user_input: str,
            response: str,
            retrieved_contexts: List[str]
    ) -> float:
        result = await self.context_utilization.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts
        )
        return result.value

    async def evaluate_all(
            self,
            user_input: str,
            response: str,
            reference: str,
            retrieved_contexts: List[str]
    ) -> Dict[str, float]:
        # Run all evaluations concurrently
        results = await asyncio.gather(
            self.evaluate_context_entity_recall(reference, retrieved_contexts),
            self.evaluate_context_precision(user_input, reference, retrieved_contexts),
            self.evaluate_context_recall(user_input, reference, retrieved_contexts),
            self.evaluate_context_utilization(user_input, response, retrieved_contexts)
        )

        return {
            "context_entity_recall": results[0],
            "context_precision": results[1],
            "context_recall": results[2],
            "context_utilization": results[3]
        }

    def print_results(self, results: Dict[str, float]) -> None:
        print("\n" + "="*60)
        print("RAGAS EVALUATION RESULTS")
        print("="*60)
        for metric, score in results.items():
            metric_name = metric.replace("_", " ").title()
            print(f"{metric_name:30s}: {score:.4f}")
        print("="*60 + "\n")


async def main():
    """
    Demo usage of the RAGASEvaluator class.
    """
    # Initialize evaluator
    evaluator = RAGContextEvaluator()

    # Example data
    user_input = "Where is the Eiffel Tower located?"
    response = "The Eiffel Tower is located in Paris."
    reference = "The Eiffel Tower is located in Paris, France."
    retrieved_contexts = [
        "The Brandenburg Gate is located in Berlin.",
        "The Eiffel Tower is located in Paris.",
        "Paris is the capital of France."
    ]

    print("Evaluating RAG system...")
    print(f"Query: {user_input}")
    print(f"Response: {response}")
    print(f"Reference: {reference}")
    print(f"Retrieved Contexts: {len(retrieved_contexts)} chunks")

    # Evaluate all metrics
    results = await evaluator.evaluate_all(
        user_input=user_input,
        response=response,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )

    # Print results
    evaluator.print_results(results)

    # Example: Evaluate individual metrics
    print("\nEvaluating individual metrics:")
    print("-" * 60)

    entity_recall = await evaluator.evaluate_context_entity_recall(
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )
    print(f"Context Entity Recall: {entity_recall:.4f}")

    precision = await evaluator.evaluate_context_precision(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )
    print(f"Context Precision: {precision:.4f}")

    recall = await evaluator.evaluate_context_recall(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )
    print(f"Context Recall: {recall:.4f}")

    utilization = await evaluator.evaluate_context_utilization(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts
    )
    print(f"Context Utilization: {utilization:.4f}")


if __name__ == "__main__":
    asyncio.run(main())