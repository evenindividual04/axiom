import asyncio

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Setup LLM and embeddings
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Supported providers: openai, google, litellm, huggingface
embeddings = embedding_factory("huggingface", model="sentence-transformers/all-MiniLM-L6-v2", client=client)

# Create metric
scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)

async def evaluate_answer_relevancy():
    # Evaluate
    result = await scorer.ascore(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967"
    )
    print(f"Answer Relevancy Score: {result.value}")

asyncio.run(evaluate_answer_relevancy())