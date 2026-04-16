from ragas.metrics import numeric_metric
from ragas.metrics.result import MetricResult
from ragas import experiment, Dataset
from ai.agents import math_agent


@numeric_metric(name="correctness")
def correctness_metric(prediction: float, actual: float):
    """Calculate correctness of the prediction."""
    if isinstance(prediction, str) and "ERROR" in prediction:
        return 0.0
    result = 1.0 if abs(prediction - actual) < 1e-5 else 0.0
    return MetricResult(value=result, reason=f"Prediction: {prediction}, Actual: {actual}")


@experiment()
async def run_experiment(row):
    expression = row["expression"]
    expected_result = row["expected"]

    # Get the model's prediction
    prediction = math_agent.run(expression)
    # print(prediction.content.result)

    try:
        # Calculate the correctness metric
        correctness = correctness_metric.score(prediction=prediction.content.result, actual=expected_result)

        return {
            "expression": expression,
            "expected_result": expected_result,
            "prediction": prediction.content.result,
            "correctness": correctness.value
        }
    except Exception as e:
        print(e)


async def main():
    dataset = Dataset.load(name="agent_eval_dataset", backend="local/csv", root_dir="..")
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
