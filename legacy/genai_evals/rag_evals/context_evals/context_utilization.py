import asyncio

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextUtilization
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = ContextUtilization(llm=llm)

async def evaluate():
    # Evaluate
    result = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[
            "The Brandenburg Gate is located in Berlin.",
            "The Eiffel Tower is located in Paris.",
        ]
    )
    print(f"Context Utilization Score: {result.value}")

asyncio.run(evaluate())