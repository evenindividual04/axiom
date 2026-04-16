# AI Evaluation Playgrounds

This collection provides evaluation frameworks for testing prompts, agents, and agentic workflows using the Ragas experimentation framework.

## Overview

Three distinct evaluation playgrounds for different AI system components:

1. **Prompt Evaluation** - Tests simple LLM prompt quality and accuracy
2. **Agent Evaluation** - Measures agent correctness for computational tasks
3. **Agentic Workflow Evaluation** - Assesses complex multi-step workflow quality

## Files

### `simple_prompt_eval_playground.py`

Evaluates basic prompt-response quality using binary pass/fail metrics.

**What it does:**
- Loads test cases from `prompt_eval_dataset.csv`
- Sends text prompts to OpenAI
- Compares predictions against expected labels
- Returns binary accuracy (pass/fail)

**Use case:** Testing classification prompts, sentiment analysis, or any task with discrete expected outputs

---

### `simple_agent_eval_playground.py`

Evaluates agent performance on mathematical expressions.

**What it does:**
- Loads math expressions from `agent_eval_dataset.csv`
- Runs a math agent to compute results
- Calculates numerical correctness (within 1e-5 tolerance)
- Handles agent errors gracefully

**Use case:** Testing computational agents, math solvers, or any agent producing numerical outputs

**Metric:** Numeric correctness with floating-point precision handling

---

### `simple_agentic_workflow_eval_playground.py`

Evaluates complex multi-step workflows using LLM-as-judge.

**What it does:**
- Loads test cases from `workflow_eval_dataset.csv`
- Executes email creation workflow
- Uses GPT-4 to judge response quality against custom criteria
- Returns pass/fail with detailed reasoning

**Use case:** Testing multi-agent systems, complex workflows, or tasks requiring qualitative assessment

**Metric:** Discrete quality metric with LLM-based evaluation

---

## Common Pattern

All three files follow the same structure:

```python
@experiment()
async def run_experiment(row):
    # 1. Execute system under test
    response = system.run(row["input"])
    
    # 2. Score the response
    score = metric.score(prediction=response, actual=row["expected"])
    
    # 3. Return structured results
    return {**row, "response": response, "score": score.value}
```

## Usage

```bash
# Run prompt evaluation
python simple_prompt_eval_playground.py

# Run agent evaluation
python simple_agent_eval_playground.py

# Run workflow evaluation
python simple_agentic_workflow_eval_playground.py
```

## Metrics Used

- **Discrete Metric** - Binary pass/fail evaluation (prompts, workflows)
- **Numeric Metric** - Precision-based scoring for numerical outputs (agents)
- **LLM-as-Judge** - GPT-4 evaluates against custom pass criteria (workflows)

## Dataset Format

Each evaluation expects CSV datasets with specific columns:

**Prompt Eval:** `text`, `label`  
**Agent Eval:** `expression`, `expected`  
**Workflow Eval:** `email`, `pass_criteria`

## Key Dependencies

- `ragas` - Experimentation and metrics framework
- `openai` - LLM API calls
- Custom modules: `ai.agents`, `ai.agentic_workflow`, `ai.call_llm`