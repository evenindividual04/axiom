import asyncio

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Create metric
scorer = Faithfulness(llm=llm)

async def evaluate_faithfulness():
    # Evaluate
    result = await scorer.ascore(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, "
            "at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )
    print(f"Faithfulness Score: {result.value}")

asyncio.run(evaluate_faithfulness())