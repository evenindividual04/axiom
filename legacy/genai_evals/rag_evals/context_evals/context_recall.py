import asyncio

from openai import AsyncOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextRecall(llm=llm)

async def evaluate():
    # Evaluate
    result = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        retrieved_contexts=["Paris is the capital of France."],
        reference="The Eiffel Tower is located in Paris."
    )
    print(f"Context Recall Score: {result.value}")

asyncio.run(evaluate())