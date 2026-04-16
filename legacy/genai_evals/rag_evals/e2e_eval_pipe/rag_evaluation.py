from typing import List

from deepeval.metrics import (
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

contextual_relevancy = ContextualRelevancyMetric(model=OllamaModel(model="granite4:3b"))
answer_relevancy = AnswerRelevancyMetric(model=OllamaModel(model="granite4:3b"))
faithfulness = FaithfulnessMetric(model=OllamaModel(model="granite4:3b"))


def evaluate_context_relevancy(user_input: str, llm_response: str, retrieved_context: List):
    test_case = LLMTestCase(
        input=user_input,
        actual_output=llm_response,
        retrieval_context=retrieved_context,
    )

    contextual_relevancy.measure(test_case)
    return {
        "metric": "context_relevancy",
        "score": contextual_relevancy.score,
        "reason": contextual_relevancy.reason,
        "response": llm_response,
        "context": retrieved_context,
    }


def evaluate_answer_relevancy(user_input: str, llm_response: str, retrieved_context: List):
    test_case = LLMTestCase(
        input=user_input,
        actual_output=llm_response,
        retrieval_context=retrieved_context,
    )

    answer_relevancy.measure(test_case)
    return {
        "metric": "answer_relevancy",
        "score": answer_relevancy.score,
        "reason": answer_relevancy.reason,
        "response": llm_response,
        "context": retrieved_context,
    }


def evaluate_answer_faithfulness(user_input: str, llm_response: str, retrieved_context: List):
    test_case = LLMTestCase(
        input=user_input,
        actual_output=llm_response,
        retrieval_context=retrieved_context,
    )

    faithfulness.measure(test_case)
    return {
        "metric": "answer_faithfulness",
        "score": faithfulness.score,
        "reason": faithfulness.reason,
        "response": llm_response,
        "context": retrieved_context,
    }
