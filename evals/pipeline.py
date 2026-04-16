from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter

from evals.failure_analysis import classify_failure
from evals.metrics import aggregate_mean, answer_relevancy, faithfulness, hallucination_rate, safety_score


@dataclass(frozen=True)
class EvalResult:
    question: str
    model: str
    prompt_version: str
    output: str
    answer_relevancy: float
    faithfulness: float
    hallucination_rate: float
    safety_score: float
    failure_type: str
    latency_ms: float
    token_count: int


def evaluate_row(
    *,
    model: str,
    prompt_version: str,
    question: str,
    context: str,
    ground_truth: str,
    output: str,
    latency_ms: float,
    token_count: int,
) -> EvalResult:
    rel = answer_relevancy(question, output)
    faithful = faithfulness(context + " " + ground_truth, output)
    halluc = hallucination_rate(context + " " + ground_truth, output)
    safe = safety_score(output)
    failure = classify_failure(output=output, ground_truth=ground_truth, context=context)

    return EvalResult(
        question=question,
        model=model,
        prompt_version=prompt_version,
        output=output,
        answer_relevancy=rel,
        faithfulness=faithful,
        hallucination_rate=halluc,
        safety_score=safe,
        failure_type=failure,
        latency_ms=latency_ms,
        token_count=token_count,
    )


def to_dict(result: EvalResult) -> dict:
    return asdict(result)


def summarize(results: list[EvalResult]) -> dict:
    if not results:
        return {
            "count": 0,
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "hallucination_rate": 0.0,
            "safety_score": 0.0,
            "latency_ms": 0.0,
            "token_count": 0.0,
            "failure_breakdown": {},
        }

    failures = Counter(result.failure_type for result in results)
    return {
        "count": len(results),
        "answer_relevancy": aggregate_mean([result.answer_relevancy for result in results]),
        "faithfulness": aggregate_mean([result.faithfulness for result in results]),
        "hallucination_rate": aggregate_mean([result.hallucination_rate for result in results]),
        "safety_score": aggregate_mean([result.safety_score for result in results]),
        "latency_ms": aggregate_mean([result.latency_ms for result in results]),
        "token_count": aggregate_mean([float(result.token_count) for result in results]),
        "failure_breakdown": dict(failures),
    }
