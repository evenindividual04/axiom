# Axiom Architecture

Detailed technical reference for Axiom's design, pipeline, and extensibility.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLI / Config Layer                          │
│                        (main.py)                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Orchestration Layer                           │
│              (src/llm_reliability_engine/orchestrator.py)        │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │   Dataset   │  │ Multi-Model  │  │  Provider Health    │    │
│  │ Generation  │─→│  Evaluation  │─→│ & Circuit Breaker   │    │
│  └─────────────┘  │    Engine    │  └─────────────────────┘    │
│                   │  (async)     │            │                 │
│                   └──────────────┘            │                 │
│                         │                     │                 │
├─────────────────────────┴─────────────────────┴─────────────────┤
│                  Evaluation Layer                               │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐ │
│  │ Per-Row Metrics│  │ Failure Class.  │  │ Aggregation &    │ │
│  │ (evals/metrics)│  │(failure_analysis)│  │ Reporting        │ │
│  └────────────────┘  └─────────────────┘  └──────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│                    Model Layer                                   │
│            (models/llm_runner.py)                               │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │Rate Limiter  │  │Retry Logic     │  │Circuit Breaker   │   │
│  │(per-provider)│  │& Exponential   │  │& Cooldown        │   │
│  │              │  │Backoff         │  │Recovery          │   │
│  └──────────────┘  └────────────────┘  └──────────────────┘   │
│      ↓                    ↓                    ↓                 │
│  ┌────────────────────────┬────────────────────────────────┐    │
│  │    Async Multi-Provider Executor                       │    │
│  │ Dispatches calls to OpenAI, Anthropic, Groq, etc.      │    │
│  └────────────────────────┬────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────┤
│                      Storage Layer                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐        │
│  │  SQLite DB  │  │  CSV Export  │  │ JSON Manifest &  │        │
│  │(runs.db)    │  │(row_results) │  │ Report (artifacts)│      │
│  └─────────────┘  └──────────────┘  └──────────────────┘        │
├──────────────────────────────────────────────────────────────────┤
│                    Presentation Layer                            │
│              (dashboard/app.py - Streamlit)                     │
└──────────────────────────────────────────────────────────────────┘
```

## Pipeline Execution Flow

### 1. Configuration Load

```python
# main.py → orchestrator.run_pipeline()
config = _load_config(config_path)

runtime_settings = {
    'concurrency': config['runtime']['concurrency'],
    'timeout_seconds': config['runtime']['request_timeout_seconds'],
    'rate_limit_per_minute': config['runtime']['rate_limit_per_minute'],
    'circuit_breaker_*': ...,  # per-provider overrides
}

models_to_eval = [m for m in config['models'] if m['enabled']]
```

### 2. Dataset Generation

```python
# data/synthetic/generate_dataset.py
dataset = generate_dataset(
    target_rows=config['dataset']['target_rows'],
    documents_dir='data/fin-docs/',  # Your domain docs
    test_type_mix=config['dataset']['test_type_mix']
)

# Returns:
# [
#   {
#     'question': str,
#     'context': str,           # Extracted from documents
#     'ground_truth': str,      # Expected correct answer
#     'test_type': str,         # 'happy_path' | 'edge_case' | 'adversarial'
#   },
#   ...
# ]
```

**Key insight**: This is where Axiom becomes domain-agnostic. You provide documents → Axiom generates Q&A pairs grounded in that domain.

### 3. Multi-Model Async Execution

```python
# models/llm_runner.py → run_models()
# Async function that dispatches to all enabled models in parallel

async def run_models(
    dataset_rows: list[dict],
    prompts: dict[str, str],          # 'base', 'improved', 'advanced'
    models: list[dict],               # Config for each model
    runtime_settings: ProviderRuntimeSettings,
) -> tuple[list[ModelOutput], list[ModelError]]:
    
    tasks = []
    for row in dataset_rows:
        for model in models:
            for prompt_version in prompts:
                # Create async task with rate limiter + circuit breaker
                task = run_single_model(
                    model, prompt_version, row, runtime_settings
                )
                tasks.append(task)
    
    # Execute all in parallel, respecting rate limits
    outputs, errors = await asyncio.gather(*tasks, return_exceptions=True)
    return outputs, errors
```

**Key mechanisms**:
- **Rate Limiter**: Enforces `rate_limit_per_minute` per provider (async queue-based)
- **Retry Logic**: Exponential backoff on transient failures (429, 500, timeout)
- **Circuit Breaker**: After `circuit_breaker_failure_threshold` failures, disable provider for `circuit_breaker_cooldown_seconds`

### 4. Per-Row Metrics & Failure Classification

```python
# evals/pipeline.py → evaluate_row()
for output in model_outputs:
    eval_result = evaluate_row(
        model=output.model,
        prompt_version=output.prompt_version,
        question=output.question,
        context=output.context,
        ground_truth=output.ground_truth,
        output=output.output,
        latency_ms=output.latency_ms,
        token_count=output.token_count,
    )
    
    # Returns EvalResult with:
    # - answer_relevancy: float (0–1)
    # - faithfulness: float (0–1)
    # - hallucination_rate: float (0–1)
    # - safety_score: float (0–1)
    # - failure_type: str (enum-like)
    # - [all inputs preserved]
```

**Metric details**:

| Metric | Type | Computation |
|--------|------|-------------|
| **Answer Relevancy** | Float [0–1] | Semantic similarity between question and output |
| **Faithfulness** | Float [0–1] | Output claims grounded in context+ground_truth |
| **Hallucination Rate** | Float [0–1] | % of output that's not in grounding documents |
| **Safety Score** | Float [0–1] | Toxicity and policy violation detection |

**Failure Classification** (evals/failure_analysis.py):

```python
def classify_failure(output: str, ground_truth: str, context: str) -> str:
    if len(output.strip()) < 10:
        return 'incomplete_answer'
    
    if _is_off_topic(output, ground_truth):
        return 'off_topic'
    
    if _hallucination_detected(output, context):
        return 'hallucination'
    
    if _timeout_in_error(output):
        return 'timeout'
    
    if _api_error_in_output(output):
        return 'error'
    
    return 'success'
```

### 5. Provider Health Tracking

```python
# src/llm_reliability_engine/orchestrator.py → ProviderHealthPolicy
class ProviderHealthPolicy:
    """
    Tracks provider state across evaluation window.
    
    State machine:
    healthy --[2+ failures in window]--> cooling_down --[timeout expires]--> healthy
                                               ↓
                                          [permanent error]
                                               ↓
                                            disabled
    """
    
    def record_errors(self, model_errors: list[ModelError]) -> None:
        for err in model_errors:
            if _is_hard_quota_error(err.error):
                # Permanently disabled
                self.state[provider] = 'disabled'
            elif _is_rate_limit_error(err.error):
                # Transient; count toward cooldown trigger
                self.transient_errors[provider] += 1
                if self.transient_errors[provider] >= threshold:
                    self.state[provider] = 'cooling_down'
                    self.disabled_until[provider] = time.monotonic() + cooldown_secs
    
    def apply(self, model_specs: list[dict]) -> list[dict]:
        # Filter out currently-disabled models
        return [
            m for m in model_specs 
            if self.get_status(m['provider']) != 'disabled'
        ]
    
    def snapshot(self) -> dict:
        # For manifest persistence
        return {
            provider: {
                'status': self.state[provider],
                'reason': self.reason[provider],
                'disabled_until': self.disabled_until[provider],
                'last_error': self.last_error[provider],
            }
            for provider in self.state
        }
```

### 6. Persistence & Artifact Generation

```python
# src/llm_reliability_engine/orchestrator.py

# 1. SQLite Persistence
_persist_run(conn, run_id, mock_mode, target_rows)
_persist_rows(conn, run_id, eval_results)

# 2. CSV Export
df = pd.DataFrame([asdict(r) for r in eval_results])
df.to_csv('experiments/row_results.csv', index=False)

# 3. JSON Manifest (full provenance)
manifest = {
    'run_id': run_id,
    'timestamp': now,
    'config_snapshot': config,
    'dataset': dataset_rows,
    'eval_results': [asdict(r) for r in eval_results],
    'provider_health_snapshot': health_policy.snapshot(),
}
write_json(MANIFESTS_DIR / f'run_{run_id}.json', manifest)

# 4. Markdown Report
report = build_run_report_markdown(
    manifest, eval_results, summarize(eval_results)
)
write_run_report(REPORTS_DIR / f'run_{run_id}.md', report)

# 5. Results Summary (points to artifacts)
results = {
    'run_id': run_id,
    'total_rows': len(dataset_rows),
    'evaluated_rows': len(eval_results),
    'success_rate': sum(1 for r in eval_results if r.failure_type == 'success') / len(eval_results),
    'manifest_path': f'experiments/manifests/run_{run_id}.json',
    'report_path': f'experiments/reports/run_{run_id}.md',
    'csv_path': 'experiments/row_results.csv',
}
write_json('experiments/results.json', results)
```

## Component Reference

### CLI Entry Point

**File**: `main.py`

Supported commands:
- `run-pipeline [--live] [--config CONFIG]`: Execute full eval pipeline
- `dashboard`: Open Streamlit UI
- `status`: Check project readiness
- `qevals-*`, `genai-*`: Legacy subcommands

### Orchestrator

**File**: `src/llm_reliability_engine/orchestrator.py`

Core logic:
- `run_pipeline()`: Main entry point
- `_load_config()`: Parse YAML config
- `_read_prompt()`: Load prompt variants
- `_ensure_schema()`: Initialize SQLite schema
- `_persist_run()`, `_persist_rows()`: Write to database
- `ProviderHealthState`, `ProviderHealthPolicy`: Health tracking state machine

### Model Runner

**File**: `models/llm_runner.py`

Key classes:
- `ModelOutput`: Result of successful call
- `ModelError`: Error from failed call
- `ProviderRuntimeSettings`: Config per provider
- `ProviderRateLimiter`: Async queue-based rate limiting (per-provider)
- `run_models()`: Main async dispatcher

Provider adapters:
- `_call_openai()`, `_call_anthropic()`, `_call_groq()`, etc.
- Each adapter handles provider-specific auth, error parsing, retry logic

### Evaluation Pipeline

**File**: `evals/pipeline.py`

- `EvalResult`: Dataclass with all metrics + metadata
- `evaluate_row()`: Computes all 4 metrics for one output
- `summarize()`: Aggregates results by model/prompt

**File**: `evals/metrics.py`

- `answer_relevancy()`: Cosine similarity (question, output)
- `faithfulness()`: Semantic entailment (context+truth, output)
- `hallucination_rate()`: Percentage not grounded in context
- `safety_score()`: LLM-based toxicity detection

**File**: `evals/failure_analysis.py`

- `classify_failure()`: Enum-like classification logic

### Dashboard

**File**: `dashboard/app.py`

Pure helper functions (testable):
- `load_artifacts()`: Reads results.json, row_results.csv, manifest
- `build_summary_cards()`: Extracts KPIs
- `build_provider_health_frame()`: DataFrame from health snapshot
- `build_delta_frame()`: Prompt comparison
- `build_failure_frame()`: Failure breakdown by type
- `format_health_status()`: State → string mapping

Main render logic:
- Hero section with run metadata
- 4 KPI cards (success rate, failures, top model, avg latency)
- 4 tabs: Overview, Prompts, Health, Artifacts
- Embedded Streamlit tables with sorting/filtering

### Reporting

**File**: `src/llm_reliability_engine/reporting.py`

- `build_run_report_markdown()`: Constructs markdown from manifest+results
- `write_run_report()`: Writes to file with latest symlink

Output:
```markdown
# Evaluation Report: run_2026-04-17_12-34-56_abc123

**Run ID**: run_2026-04-17_12-34-56_abc123
**Timestamp**: 2026-04-17 12:34:56 UTC
**Models Evaluated**: gpt-4o-mini, claude-3-5-haiku, gemini-2.0-flash

## Summary

- **Total Rows**: 100
- **Evaluated**: 98 (success), 2 (error)
- **Success Rate**: 98%
- **Avg Answer Relevancy**: 0.92
- **Avg Faithfulness**: 0.88
- **Avg Hallucination Rate**: 0.05
- **Avg Safety Score**: 0.95

## Failures

| Model | Prompt | Count | Type |
|-------|--------|-------|------|
| gpt-4o-mini | base | 1 | timeout |
| claude-3-5-haiku | improved | 1 | hallucination |

[Full failure details with examples...]

## Provider Health

| Provider | Status | Reason | Expires |
|----------|--------|--------|---------|
| openai | healthy | - | - |
| anthropic | cooling_down | Rate limit | 2026-04-17 12:36:56 |
| groq | healthy | - | - |

...
```

## Data Flow Diagram

```
Config (YAML)
    ↓
CLI (main.py)
    ↓
Orchestrator.run_pipeline()
    ├─ Load config
    ├─ Generate dataset
    │   └─ Read documents (data/fin-docs/)
    │       ↓
    │   Create Q&A pairs
    │
    ├─ Run models (async)
    │   └─ Rate limiter
    │   └─ Retry logic
    │   └─ Circuit breaker
    │   └─ Provider health tracking
    │       ↓
    │   [ModelOutput] + [ModelError]
    │
    ├─ Evaluate rows
    │   └─ Metrics (relevancy, faithfulness, hallucination, safety)
    │   └─ Failure classification
    │       ↓
    │   [EvalResult] per row
    │
    └─ Persist & emit artifacts
        ├─ SQLite (runs.db)
        ├─ CSV (row_results.csv)
        ├─ JSON Manifest (experiments/manifests/)
        ├─ Markdown Report (experiments/reports/)
        └─ Results Summary (results.json)
                ↓
            Dashboard (reads results.json + artifacts)
```

## Configuration Hierarchy

```yaml
config.yaml (base)
    ├─ dataset
    │   ├─ target_rows
    │   ├─ min_rows, max_rows
    │   └─ test_type_mix (happy_path, edge_case, adversarial)
    │
    ├─ models[] (list)
    │   ├─ name (e.g., "gpt-4o-mini")
    │   ├─ provider (e.g., "openai")
    │   └─ enabled (bool)
    │
    └─ runtime
        ├─ mock_mode
        ├─ mock_fallback_on_failure
        ├─ concurrency
        ├─ request_timeout_seconds
        ├─ rate_limit_per_minute (global default)
        ├─ max_retries
        ├─ retry_base_delay_seconds
        ├─ retry_max_delay_seconds
        ├─ circuit_breaker_failure_threshold
        ├─ circuit_breaker_cooldown_seconds
        │
        └─ provider_overrides (per-provider tuning)
            ├─ openai
            ├─ anthropic
            ├─ groq
            ├─ gemini
            ├─ openrouter
            └─ [...]
```

## Extensibility Points

### Adding a Custom Metric

1. **File**: `evals/metrics.py`
   ```python
   def my_domain_metric(output: str, context: str) -> float:
       # Your evaluation logic
       return score
   ```

2. **File**: `evals/pipeline.py`
   ```python
   def evaluate_row(...) -> EvalResult:
       # ... existing metrics
       custom = my_domain_metric(output, context)
       return EvalResult(..., custom_score=custom)
   ```

3. **File**: `evals/pipeline.py` (update EvalResult dataclass)
   ```python
   @dataclass(frozen=True)
   class EvalResult:
       # ... existing fields
       custom_score: float
   ```

### Adding a Provider

1. **File**: `models/llm_runner.py`
   ```python
   async def _call_my_provider(
       model: str,
       prompt: str,
       question: str,
       context: str,
       ground_truth: str,
       api_key: str,
       settings: ProviderRuntimeSettings,
   ) -> ModelOutput:
       # Your provider-specific logic
       pass
   ```

2. **File**: `models/llm_runner.py` (in `run_models()`)
   ```python
   elif output.provider == 'my_provider':
       return await _call_my_provider(...)
   ```

3. **File**: `config.yaml`
   ```yaml
   models:
     - name: my-model
       provider: my_provider
       enabled: true
   ```

4. **File**: `.env`
   ```bash
   MY_PROVIDER_API_KEY=...
   ```

### Customizing Dataset Generation

**File**: `data/synthetic/generate_dataset.py`

Modify `generate_dataset()` to:
- Change Q&A generation strategy
- Use different document chunking
- Add domain-specific test patterns

```python
def generate_dataset(
    target_rows: int,
    documents_dir: str,
    test_type_mix: dict,
) -> list[dict]:
    # Your custom logic
    pass
```

### Adding Domain-Specific Failure Types

**File**: `evals/failure_analysis.py`

```python
def classify_failure(output: str, ground_truth: str, context: str) -> str:
    if _domain_specific_check(output):
        return 'domain_specific_failure'
    
    # ... existing checks
    return 'success'
```

## Testing Strategy

**Test Coverage**: 82 unit tests, 52% overall, 47% critical path

**Test Categories**:

1. **Model Layer** (`tests/test_circuit_breaker.py`, `tests/test_rate_limiter.py`, `test_provider_health_policy.py`)
   - Rate limiter: Window sliding, rejection on limit exceed
   - Circuit breaker: State transitions, cooldown recovery
   - Health policy: Permanent vs temporary failures, cooldown expiry

2. **Eval Layer** (`tests/test_metrics.py`, `test_failure_analysis.py`)
   - Per-metric correctness (relevancy, faithfulness, etc.)
   - Failure classification logic

3. **Orchestration** (`tests/test_orchestrator_manifest.py`)
   - Manifest generation and artifact paths
   - Config loading
   - Run persistence

4. **Dashboard** (`tests/test_dashboard.py`)
   - Helper function purity (no side effects)
   - Data transformation correctness

5. **Integration** (E2E runs in CI)
   - Full pipeline execution (mock mode)
   - Artifact emission

## Performance Considerations

### Concurrency & Latency

- **Async model execution**: Uses `asyncio` with `tasks = await asyncio.gather(...)`
- **Typical latency**: 100 rows × 3 models ≈ 2–5 minutes (depends on parallelism, provider latency)
- **Tuning**: Adjust `runtime.concurrency` in config (default 3)

### Rate Limiting

- **Per-provider**: Each provider gets own `ProviderRateLimiter` instance
- **Sliding window**: `rate_limit_per_minute` enforced via async queue
- **Tuning**: Adjust per-provider in `config.yaml` → `runtime.provider_overrides`

### Database

- **SQLite**: Single-threaded; safe for concurrent reads, single writer (Python CLI only)
- **Indexes**: (run_id, model, prompt_version, question) for dashboard queries
- **Scaling**: For 1000s of runs, consider PostgreSQL (future work)

### Memory

- **Dataset**: 100 rows typically ≈ 100KB–1MB (depends on document chunk size)
- **Results**: Similar; streamed to CSV as written
- **Async tasks**: `concurrency=3` keeps memory footprint low

## Error Handling & Resilience

### Transient Errors (Retryable)

```
429 (Rate Limit) ────→ exponential backoff
500 (Server Error)     max_retries attempts
Timeout (>60s)         base_delay + jitter
```

### Permanent Errors (Circuit Breaker)

```
401 (Unauthorized) ──→ disabled (manual fix)
403 (Forbidden)
429 (After max retries)
quota_exceeded         permanent disable
```

### Partial Failure Handling

- **Per-row failures**: Logged, not fatal. Eval continues.
- **Provider failures**: Circuit breaker disables provider for cooldown.
- **Mock fallback**: If `mock_fallback_on_failure=true`, use synthetic response instead of crashing.

## Debugging Tips

### Enable Verbose Logging

```python
# In models/llm_runner.py or evals/pipeline.py
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

### Inspect SQLite Database

```bash
sqlite3 experiments/runs.db

# List tables
.tables

# Query runs
SELECT run_id, created_at FROM runs ORDER BY created_at DESC LIMIT 5;

# Query eval results
SELECT model, prompt_version, failure_type, COUNT(*) 
FROM eval_rows 
WHERE run_id = 'run_2026-04-17_12-34-56_abc123'
GROUP BY model, prompt_version, failure_type;
```

### Trace Async Execution

```bash
# In models/llm_runner.py
asyncio.run(main(), debug=True)
```

### Check Provider Health

```bash
# Extract from manifest
cat experiments/manifests/run_*.json | jq '.provider_health_snapshot'
```

## Future Work

- [ ] PostgreSQL backend for multi-instance deployments
- [ ] Distributed evaluation (Celery, Ray)
- [ ] Real-time streaming dashboard (WebSocket)
- [ ] Custom metric plugin system
- [ ] Provider-agnostic standardized error codes
- [ ] Batch evaluation scheduling (cron/APScheduler)
