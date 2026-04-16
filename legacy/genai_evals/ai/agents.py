from agno.agent import Agent
from agno.models.anthropic import Claude
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

llm = Claude(id="claude-sonnet-4-5", temperature=0.5)

SYSTEM_MESSAGE = """You are a mathematical problem-solving agent.

Your task is to break down complex mathematical expressions into a sequence of atomic operations, following proper order of operations.

When you have the final answer, respond with just the number."""

class MathResponse(BaseModel):
    result: float | int = Field(..., description="The final result of the math problem.")


math_agent = Agent(
    name="math agent",
    description="You are university math professor-X, proficient at solving advanced math problems.",
    system_message=SYSTEM_MESSAGE,
    instructions="Always solve math problem given to you and just give the answer to the problem nothing else. "
                 "Always convert fractions to floating point result with 2 decimals."
                 "The final answer should only be integer or floating point numbers not strings."
                 "Bad Example: Answer = 2.83 or Result = 10 not accepted."
                 "Good Examples 2.83, 10.",
    model=llm,
    debug_mode=False,
    output_schema=MathResponse
)