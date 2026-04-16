import os
from agno.agent import Agent
from agno.eval import PerformanceEval
from agno.models.anthropic import Claude
from dotenv import load_dotenv, find_dotenv
from tool import retrieve_leukemia_knowledge_base
from prompts import SYSTEM_PROMPT

load_dotenv(find_dotenv())

claude_llm = Claude(id="claude-sonnet-4-20250514", temperature=0.7, api_key=os.environ.get("ANTHROPIC_API_KEY"))

def run_agent():
    # Instantiate the Agno Agent with the knowledge base
    _agent = Agent(
        model=claude_llm,
        tools=[retrieve_leukemia_knowledge_base],
        debug_mode=True,
        system_message=SYSTEM_PROMPT,
        instructions="Always give the response in bullets or tabular format."
    )
    response = _agent.run("How does the blast percentage differ between acute and chronic leukemia?")
    print(f"Agent response: {response.content}")

    return response

simple_response_perf = PerformanceEval(
    name="Simple Performance Evaluation",
    func=run_agent,
    num_iterations=3,
    warmup_runs=0,
)

if __name__ == "__main__":
    simple_response_perf.run(print_results=True, print_summary=True)