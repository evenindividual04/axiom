# Axiom

**Sector-agnostic LLM reliability evaluation engine for production AI systems.**

Axiom runs LLM agents through dynamic synthetic test suites grounded in your domain data, evaluates responses across 7+ model providers, classifies failures, and tracks provider health with automatic cooldown recovery. Each run produces a full manifest, markdown report, and live dashboard.

Use Axiom for **finance, healthcare, legal, e-commerce, customer support, code generation**—or any domain where reliable LLM outputs matter.

## Why Axiom?

Building reliable LLM systems requires:
1. **Exhaustive evaluation.** Not just accuracy—answer relevancy, faithfulness, hallucination rate, safety. Axiom tracks all four, plus custom failure classification.
2. **Real-world data.** Synthetic tests grounded in actual domain documents. No fake prompts.
3. **Multi-model validation.** OpenAI, Anthropic, Groq, Gemini, OpenRouter, Ollama, ZAI. See which models survive pressure in your domain.
4. **Provider failure is expected.** Rate limits, quota exhaustion, timeouts. Axiom includes automatic circuit breaker with cooldown recovery so one failing provider doesn't crash your evals.
5. **Full artifact trail.** Every run generates a manifest (provenance), markdown report (human-readable summary), and structured CSV (data science). Reproducible from day one.

## Quick Start

### 1. Setup

```bash
git clone https://github.com/evenindividual04/axiom.git
cd axiom

# Install dependencies and set up environment
bash scripts/bootstrap.sh

# Verify setup
uv run python main.py status
```

### 2. Prepare Your Domain Data

Provide domain documents in `data/fin-docs/` (we kept the folder name for backward compatibility, but use it for any domain):

```bash
# Place your domain documents here
cp /path/to/your/docs/* data/fin-docs/

# Supported formats: .txt, .md, .pdf
# The dataset generator automatically ingests both data/raw/ and data/fin-docs/
```

Examples:
- **Finance**: SEC filings, annual reports, financial statements
- **Healthcare**: Clinical guidelines, medical literature, treatment protocols
- **Legal**: Case law, contracts, regulatory documents
- **E-commerce**: Product catalogs, policies, FAQ documents
- **Customer support**: Knowledge base articles, troubleshooting guides

### 3. Configure Providers

Edit `config.yaml` to enable models and set runtime options:

```yaml
models:
  - name: gpt-4o-mini
    provider: openai
    enabled: true
  - name: claude-3-5-haiku-latest
    provider: anthropic
    enabled: true
  - name: gemini-2.0-flash
    provider: gemini
    enabled: true

runtime:
  mock_mode: false
  concurrency: 3
  request_timeout_seconds: 60
  circuit_breaker_cooldown_seconds: 120
  provider_overrides:
    openai:
      rate_limit_per_minute: 6
      max_retries: 2
```

Set provider API keys in `.env`:
```bash
cp .env.example .env
# Edit .env with your API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk-..."
# etc.
```

### 4. Run Full Pipeline

```bash
uv run python main.py run-pipeline --live
```

This executes:
- **Data generation**: Synthetic test suite with grounded financial documents (100+ rows by default)
- **Multi-model evaluation**: All enabled models run all test cases
- **Metrics computation**: Answer relevancy, faithfulness, hallucination, safety per row
- **Failure classification**: Proprietary logic labels failures (incomplete_answer, off_topic, hallucination, etc.)
- **Provider health tracking**: Circuit breaker logs rate limits, quota exhaustion, timeouts
- **Artifact emission**: Manifest, report, CSV, JSON, database

Output files:
```
experiments/
├── manifests/
│   └── run_<timestamp>_<uuid>.json        # Full evaluation metadata
├── reports/
│   └── run_<timestamp>_<uuid>.md          # Human-readable markdown
├── row_results.csv                        # Raw per-row metrics
├── results.json                           # Summary with artifact links
└── runs.db                                # SQLite persistence
```

### 5. Explore Results

```bash
uv run python main.py dashboard
```

Opens Streamlit dashboard with:
- **Overview**: Run KPIs, success rate, failure breakdown by type
- **Prompts**: Baseline vs improved prompt deltas, sorted by measurable improvement
- **Provider Health**: Current state (healthy/cooling_down/disabled) with reason and cooldown expiry
- **Artifacts**: Direct links to latest manifest, report, CSV

## Architecture

### Evaluation Pipeline

```
Data Generation
    ↓
Multi-Model Async Execution (with circuit breaker)
    ↓
Per-Row Metrics (relevancy, faithfulness, hallucination, safety)
    ↓
Failure Classification
    ↓
Provider Health Policy (cooldown recovery)
    ↓
Manifest + Report + CSV Emission
```

### Provider Health & Resilience

Axiom tracks three error types:

| Error Type | Behavior | Recovery |
|---|---|---|
| **Rate limit** | Temporary; provider pauses | Auto: cooldown period, then retry |
| **Quota exhausted** | Caller's fault; provider disabled | Manual: fix API quota, re-enable in config |
| **System error** | Transient; circuit breaker triggers | Auto: cooldown threshold, then healthy |

Health state machine:
```
healthy
  ↓ (2+ failures in window)
  ↓
cooling_down (120s default)
  ↓ (cooldown expires)
  ↓
healthy
```

Config-disabled models stay disabled regardless of errors.

### Metrics

**Per-row evaluation:**
- **Answer Relevancy**: Does output address the question? (0–1)
- **Faithfulness**: Does output align with grounded context? (0–1)
- **Hallucination Rate**: Hallucinated claims as % of output. (0–1)
- **Safety Score**: Output toxicity and policy violations. (0–1)
- **Failure Type**: Proprietary classifier (incomplete_answer, off_topic, hallucination, timeout, error, success)

**Aggregates:**
- Success rate (% of rows with no failure)
- Mean metrics by model and prompt version
- Latency p50, token count by provider

### Failure Classification

Custom logic in `evals/failure_analysis.py` examines output text and ground truth to label:
- **incomplete_answer**: Output too short or obviously cut off
- **off_topic**: Output addresses wrong domain
- **hallucination**: Factual claim not in context
- **timeout**: Model exceeded request_timeout_seconds
- **error**: Provider API error (rate limit, quota, system)
- **success**: No failure detected

## Configuration Reference

Full config in `config.yaml`:

```yaml
dataset:
  target_rows: 100                    # Target test suite size
  min_rows: 200                       # Min synthetic test cases
  max_rows: 500                       # Max synthetic test cases
  test_type_mix:
    happy_path: 0.55                  # 55% normal cases
    edge_case: 0.30                   # 30% boundary cases
    adversarial: 0.15                 # 15% adversarial cases

runtime:
  mock_mode: false                    # Use mock responses (dev only)
  mock_fallback_on_failure: true      # Fall back to mock if live fails
  concurrency: 3                      # Parallel async requests
  request_timeout_seconds: 60         # Timeout per request
  rate_limit_per_minute: 6            # Global rate limit
  max_retries: 2                      # Retry count
  circuit_breaker_failure_threshold: 2  # Failures before cooldown
  circuit_breaker_cooldown_seconds: 120 # Cooldown duration

  provider_overrides:
    openai:
      rate_limit_per_minute: 6
      circuit_breaker_cooldown_seconds: 120
    anthropic:
      rate_limit_per_minute: 3
      circuit_breaker_cooldown_seconds: 240
```

## Testing & Quality

Axiom includes 82 unit tests covering:
- Multi-model async execution
- Circuit breaker state transitions
- Rate limiter accuracy
- Provider health cooldown recovery
- Dashboard helpers and artifact loading
- Manifest and report generation
- End-to-end pipeline validation

Coverage: 52.09% overall, 47.72% critical path (gates at 47%/42%).

Run tests:
```bash
pytest -v
pytest --cov=src/llm_reliability_engine --cov=evals --cov=models
```

CI/CD: GitHub Actions on every push (lint + test + coverage gates).

## Project Structure

```
axiom/
├── config.yaml                        # Runtime config
├── main.py                            # CLI entry point
├── pyproject.toml                     # Dependencies
├── README.md                          # This file
│
├── data/
│   ├── raw/                          # Seed domain documents
│   ├── synthetic/                    # Generated test suites
│   └── fin-docs/                     # User-provided domain documents
│
├── models/
│   ├── llm_runner.py                # Multi-provider async execution
│   └── __init__.py
│
├── evals/
│   ├── pipeline.py                  # Evaluation row logic
│   ├── metrics.py                   # Metric functions
│   ├── failure_analysis.py          # Failure classification
│   └── __init__.py
│
├── prompts/
│   ├── base_prompt.txt
│   ├── improved_prompt.txt
│   └── advanced_prompt.txt
│
├── src/llm_reliability_engine/
│   ├── orchestrator.py              # Pipeline orchestration
│   ├── reporting.py                 # Report generation
│   └── __init__.py
│
├── dashboard/
│   ├── app.py                       # Streamlit UI
│   └── __init__.py
│
├── tests/
│   ├── test_orchestrator_manifest.py
│   ├── test_provider_health_policy.py
│   ├── test_dashboard.py
│   ├── test_circuit_breaker.py
│   └── ... (82 tests total)
│
├── scripts/
│   ├── bootstrap.sh                 # Setup script
│   └── download_seed_docs.py        # Refresh financial docs
│
└── experiments/
    ├── manifests/                   # Run provenance
    ├── reports/                     # Markdown summaries
    ├── results.json                 # Latest summary
    ├── row_results.csv              # Raw metrics
    └── runs.db                      # SQLite store
```

## Advanced Usage

### Customizing Domain Data

Axiom is built to work with any domain. Here's how:

1. **Replace seed documents**: `scripts/seed_doc_sources.json` contains URLs for the default financial documents. Modify to pull from your domain's sources (regulatory bodies, knowledge bases, databases, etc.).

   ```json
   {
     "sources": [
       "https://your-healthcare-org.com/guidelines/",
       "https://your-legal-db.com/cases/"
     ]
   }
   ```

2. **Refresh docs**:
   ```bash
   uv run python scripts/download_seed_docs.py
   ```

3. **Or upload manually**: Copy your domain documents directly to `data/fin-docs/` (or rename the folder if you prefer). Supported: `.txt`, `.md`, `.pdf`.

4. **Customize test prompts**: Edit `prompts/*.txt` to match your domain's questions:

   **Finance example:**
   ```
   Analyze the risk factors in this 10-K filing. What are the top 3 risks to shareholder value?
   ```

   **Healthcare example:**
   ```
   Based on the clinical guidelines, what is the recommended treatment for this patient presentation?
   ```

   **Legal example:**
   ```
   Summarize the precedent set by this case for contract interpretation.
   ```

### Prompt Versions

Axiom compares multiple prompts. Edit `prompts/` and load them in orchestrator:
```python
version = _read_prompt("base")      # base_prompt.txt
version = _read_prompt("improved")  # improved_prompt.txt
```

Dashboard shows deltas: which prompt version improves hallucination, relevancy, etc.?

### Custom Metrics

Add metrics to `evals/metrics.py` and integrate in `evals/pipeline.py`:
```python
def my_custom_metric(output: str, context: str) -> float:
    # Your logic
    return score

# In evaluate_row():
custom = my_custom_metric(output, context)
```

### Mock Mode (Development)

For fast iteration without API costs:
```bash
uv run python main.py run-pipeline --mock
```

Generates synthetic responses instead of calling providers. Set `mock_fallback_on_failure: true` in config to use mock as fallback when live requests fail.

## Troubleshooting

### "API key not found"
Ensure `.env` is populated and in project root:
```bash
echo "OPENAI_API_KEY=sk-..." >> .env
```

### "Provider cooling down"
Provider hit rate limit or failure threshold. Check dashboard **Provider Health** tab:
- **healthy**: Ready for requests
- **cooling_down**: Will retry after cooldown expires (default 120s)
- **disabled**: Manual fix needed (check API quota or permissions)

### "Tests failing"
```bash
pytest -v tests/test_orchestrator_manifest.py
pytest --pdb              # Drop into debugger on failure
```

### "Dashboard not loading"
```bash
uv run python main.py dashboard
# Check console for errors; common: missing experiments/results.json
```

## Performance Notes

- **Concurrency**: Set `runtime.concurrency` to balance throughput vs. rate limit compliance. Default 3.
- **Latency**: Axiom routes requests async; end-to-end time depends on parallelism and provider response times. Typical 100-row suite: 2–5 minutes.
- **SQLite store**: Axes on (run_id, model, prompt_version, question). Fast for dashboard queries. Safe for concurrent reads (single writer via Python asyncio loop).

## License

MIT

## Contributing

Issues and PRs welcome. Please:
1. Add tests for new features (target 80%+ coverage)
2. Run `ruff check .` and `pytest` before submitting
3. Update config examples and README if adding new config keys
