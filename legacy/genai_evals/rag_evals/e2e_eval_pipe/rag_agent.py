import os
from typing import Optional

from agno.eval import ReliabilityEval, ReliabilityResult
# from agno.utils.pprint import pprint_run_response
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from dotenv import load_dotenv, find_dotenv

from rag_evaluation import evaluate_context_relevancy, evaluate_answer_relevancy, evaluate_answer_faithfulness
from tool import retrieve_leukemia_knowledge_base
from prompts import SYSTEM_PROMPT

load_dotenv(find_dotenv())

#gemini_llm = Gemini(id="gemini-2.5-flash", temperature=0.7, api_key=os.environ.get("GEMINI_API_KEY"))
claude_llm = Claude(id="claude-sonnet-4-20250514", temperature=0.7, api_key=os.environ.get("ANTHROPIC_API_KEY"))
# gpt_llm = OpenAIChat(id="gpt-4o", temperature=0.7, api_key=os.environ.get("OPENAI_API_KEY"))
# llm = Ollama(id='gemma3:12b')

def evaluate_agent_reliability(llm_response):
    evaluation = ReliabilityEval(
        name="Tool Call Reliability",
        agent_response=llm_response,
        expected_tool_calls=["retrieve_leukemia_knowledge_base"],
    )
    result: Optional[ReliabilityResult] = evaluation.run(print_results=True)
    print(result)

def create_claude_agent():
    # Instantiate the Agno Agent with the knowledge base
    _agent = Agent(
        model=claude_llm,
        tools=[retrieve_leukemia_knowledge_base],
        debug_mode=True,
        system_message=SYSTEM_PROMPT,
        instructions="Always give the response in bullets or tabular format."
    )
    return _agent

agent = create_claude_agent()

query = "How does the blast percentage differ between acute and chronic leukemia?"
# query = "What chromosomal translocation defines CML, and what genes are involved?"
# query = "What is acute respiratory syndrome?"
# agent.print_response(input=query)

response = agent.run(query)
# collect context and response from agent
agent_response = response.content
context = []
if response.tools:
    for tool in response.tools:
        context.append(tool.result)

# start the rag evals
context_relevancy = evaluate_context_relevancy(user_input=query, retrieved_context=context, llm_response=agent_response)
print(context_relevancy)
answer_relevancy = evaluate_answer_relevancy(user_input=query, retrieved_context=context, llm_response=agent_response)
print(answer_relevancy)
faithfulness = evaluate_answer_faithfulness(user_input=query, retrieved_context=context, llm_response=agent_response)
print(faithfulness)

# start agent evals
evaluate_agent_reliability(llm_response=response)



