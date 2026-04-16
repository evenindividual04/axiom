import os

from openai import OpenAI
from ragas.metrics import DiscreteMetric
from ragas import experiment, Dataset
from ragas.llms import llm_factory
from ai.agentic_workflow import email_creation_workflow
from dotenv import load_dotenv, find_dotenv

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm = llm_factory("gpt-4o", client=openai_client)

my_metric = DiscreteMetric(
    name="response_quality",
    prompt="Evaluate the response based on the pass criteria: {pass_criteria}. Does the response meet the criteria? Return 'pass' or 'fail'.\nResponse: {response}",
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    response = email_creation_workflow.run(row["email"])

    score = my_metric.score(
        llm=llm,
        response=response,
        pass_criteria=row["pass_criteria"]
    )

    experiment_view = {
        **row,
        "response": response,
        "score": score.value,
        "score_reason": score.reason,
    }
    return experiment_view

async def main():
    dataset = Dataset.load(name="workflow_eval_dataset", backend="local/csv", root_dir="/Users/pavanmantha/Pavans/PracticeExamples/DataScience_Practice/genai_evals")
    experiment_result = await run_experiment.arun(dataset)
    print("Experiment_result: ", experiment_result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())