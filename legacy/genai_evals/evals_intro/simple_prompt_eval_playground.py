from ragas.metrics import discrete_metric
from ragas.metrics.result import MetricResult
from ragas import experiment, Dataset
from ai.call_llm import trigger_openai

@discrete_metric(name="accuracy", allowed_values=["pass", "fail"])
def my_metric(prediction: str, actual: str):
    """Calculate accuracy of the prediction."""
    return MetricResult(value="pass", reason="") if prediction == actual else MetricResult(value="fail", reason="")

@experiment()
async def run_experiment(row):

    response = trigger_openai(row["text"])
    score = my_metric.score(
        prediction=response,
        actual=row["label"]
    )

    experiment_view = {
        **row,
        "response":response,
        "score":score.value,
    }
    return experiment_view

async def main():
    dataset = Dataset.load(name="prompt_eval_dataset", backend="local/csv", root_dir="..")
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())