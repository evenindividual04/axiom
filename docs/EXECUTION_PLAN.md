# LLM Reliability Engine - Detailed Execution Plan

## Objective
Build a production-style reliability evaluation engine for financial AI systems by combining `genai_evals` and `qevals` into one coherent project, with dynamic synthetic data generation, multi-model evaluation, and measurable prompt improvement.

## Scope Guardrails
1. Do not use a static downloaded CSV as the core dataset.
2. Use dynamic synthetic generation from seeded financial documents.
3. Prioritize end-to-end execution in pure Python (`asyncio`) before adding complexity.
4. Keep original workspace folders untouched; all edits happen in `llm-reliability-engine`.
5. Reuse only required capabilities from legacy code (`qevals` datagen/eval ideas and `genai_evals` metric patterns), not full legacy structure in runtime flow.
6. Use UV as the primary environment and dependency manager.

## Build First vs Skip

### Build First (must ship)
1. Synthetic dataset generator producing 200-500 rows in `question,context,ground_truth` format.
2. Evaluation pipeline with answer relevancy, faithfulness, and hallucination metrics.
3. Multi-model runner (GPT + optional Claude + one open-source endpoint).
4. Failure classification layer (`numerical_error`, `hallucination`, `ambiguity`, `compliance_risk`, `correct`).
5. Prompt optimization loop with baseline vs improved prompts.
6. JSON metrics output for experiments.

### Skip Until Core Is Stable
1. Go/Python service boundary.
2. Distributed orchestration infra.
3. Fine-tuning and advanced model training.
4. Heavy frontend beyond lightweight Streamlit.

## Modern Framework Choices
1. `uv` for reproducible Python environment and dependency sync.
2. `streamlit` for fast analytics dashboard delivery.
3. `mlflow` for prompt-version experiment tracking.
4. `sqlite` for local structured run storage.

## Provider Strategy
1. Support multiple live providers through a unified runner interface:
   - OpenAI
   - Groq
   - Gemini
   - OpenRouter
   - Z.ai
   - Anthropic (optional)
   - Ollama (open-source local)
2. Final deliverable remains live-first (`mock_mode=false`).
3. Keep `mock_fallback_on_failure=true` as emergency continuity mode only when live calls fail completely.

## Runtime Project Shape (Target)
```text
llm-reliability-engine/
   data/
      raw/
      synthetic/
   evals/
      pipeline.py
      metrics.py
      failure_analysis.py
   models/
      llm_runner.py
   prompts/
      base_prompt.txt
      improved_prompt.txt
   experiments/
      results.json
   main.py
   config.yaml
```

This target shape is the primary runtime surface. `legacy/` remains available as reference and extraction source only.

## Plug-and-Play Asset Pack (Execution Inputs)

### Dataset Seed Kit
1. Include explicit financial seeds across:
   - loans and EMI
   - insurance claims and exclusions
   - risk and adversarial abuse attempts
   - numerical reasoning
2. Normalize every generated row to:
   - `question`
   - `context`
   - `ground_truth`
3. Expand seed set programmatically to 200+ rows via synthetic variants.

### Prompt Kit
1. Base prompt (intentionally weaker baseline).
2. Improved prompt (factual, anti-hallucination, safe behavior).
3. Advanced prompt (optional, structured answer + reasoning).

### Metric Kit
1. Faithfulness
2. Answer relevancy
3. Hallucination rate
4. Safety score

### Failure Mapping Kit
Use fixed labels for downstream analytics:
1. `numerical_error`
2. `hallucination`
3. `ambiguity`
4. `compliance_risk`
5. `correct`

### Output Artifact Kit
1. Baseline vs improved summary JSON in `experiments/results.json`.
2. Model-level comparison table for prompt versions.

## Phase 0: Base Integration (Completed)
1. Create single project root and copy both repos under `legacy/`.
2. Add unified runner, packaging, bootstrap, and validation scripts.
3. Apply minimum compatibility fixes for copied `qevals` execution.

Deliverable:
- One runnable project shell with preserved originals.

## Phase 1: Dynamic Dataset Pipeline (Day 1-2)
1. Curate 5-10 public financial seed documents spanning lending, payments, insurance, and investments.
2. Ingest seed documents through `qevals` data pipeline.
3. Generate 200-500 synthetic cases with strict schema:
   - `question`
   - `context`
   - `ground_truth`
   - `test_type` (`happy_path`, `edge_case`, `adversarial`)
4. Enforce explicit dataset mix targets:
   - 50-60% happy path
   - 25-35% edge case multi-hop
   - 10-20% adversarial and bias probes

Deliverable:
- Versioned synthetic dataset in `data/synthetic/` with type-balanced coverage.

## Phase 2: Async Orchestration Layer (Day 2-3)
1. Build central orchestrator in Python using `asyncio`.
2. Run concurrent inference across target models:
   - OpenAI (GPT-4o-mini)
   - Anthropic (Haiku, optional)
   - One open-source model endpoint
3. Capture per request:
   - prompt text
   - model output
   - latency
   - token usage (prompt/completion/total when available)

Deliverable:
- Non-sequential execution engine with complete run metadata.

## Phase 3: Evaluation Layer Integration (Day 3-4)
1. Use `genai_evals` metric modules to score outputs.
2. Mandatory metrics:
   - faithfulness / groundedness
   - answer relevancy
   - hallucination rate
3. Add custom failure classification:
   - `numerical_error`
   - `hallucination`
   - `ambiguity`
   - `compliance_risk`
   - `correct`

Deliverable:
- Structured evaluation output per sample and aggregated per model.

## Phase 4: Logging, Experiment Tracking, and Prompt Loop (Day 4-5)
1. Persist all runs to SQLite (or Postgres) with normalized tables for:
   - datasets
   - model_runs
   - metrics
   - failure_labels
2. Integrate MLflow run tracking for prompt versioning and metric drift.
3. Implement prompt optimization loop:
   - evaluate baseline prompt
   - evaluate improved prompt(s)
   - compare and select best by weighted metric objective

Deliverable:
- Reproducible paper trail from prompt version to metric improvement.

## Phase 5: Visualization and Narrative (Day 5-7)
1. Build Streamlit dashboard for:
   - model comparison
   - prompt comparison
   - latency distribution
   - failure-type breakdown
2. Keep visual style high signal and interview-ready.
3. Export machine-readable summary JSON in `experiments/results.json`.

Deliverable:
- Decision-grade UI plus artifact files suitable for demos and interviews.

## Validation Gates
1. Compile and import checks pass.
2. Dataset generation yields all three test types.
3. Async runner processes batch without sequential bottlenecks.
4. Metrics and failure labels are persisted to DB.
5. Prompt loop shows measurable deltas between baseline and improved prompts.

## Final Product Gate (No Mock Deliverable)
1. `mock_mode` must be set to `false` for final reported runs.
2. At least 200 generated rows must be evaluated end-to-end in one run.
3. Run must execute against at least 2 live models (GPT + one additional provider).
4. Prompt versions `base` and `improved` must both run and be compared.
5. `experiments/results.json` must contain baseline vs improved aggregates per model.
6. Run artifacts must include per-sample logs with latency and token counts.
7. SQLite run store must persist datasets, outputs, scores, and failure labels.
8. README quickstart must include one command for dataset generation and one for full evaluation.
9. Final report must include hallucination delta and faithfulness delta from live runs only.

## Risks To Close Before Final Submission
1. Real provider credential management and failure-safe fallback behavior.
2. Seed data provenance documentation (public links and license notes).
3. Deterministic reproducibility controls (fixed seed, config snapshot, run ID).
4. Basic test coverage for dataset generation, metric functions, and failure classifier.

## 10-12 Day Constraint Strategy
1. First 4 days: ship complete pipeline without dashboard polish.
2. Next 2-3 days: improve prompt loop and analysis quality.
3. Final 2-3 days: dashboard polish, README narrative, and demo hardening.
4. Avoid Go/Python split unless core Python flow is already stable.

## Success Criteria
1. End-to-end flow is reproducible from one project root.
2. Dataset is dynamically generated, domain-grounded, and edge-case aware.
3. Model outputs are scored with reliability metrics and classified failures.
4. Prompt updates produce quantified reliability improvements.
5. Results are visible in both JSON artifacts and dashboard views.

## Remaining Implementation Plan (Seven Workstreams)

### Workstream 1: Automated Verification Hardening
**Goal:** Protect the runtime paths that currently make or break the repo: live-provider execution, fallback behavior, progress reporting, and summary generation.

**Why this comes first:** The pipeline already runs, but the most fragile behaviors are not deeply protected. Without tests here, any later change can quietly regress the live demo.

**Scope:**
1. Add unit tests for `models/llm_runner.py` covering:
   - retryable vs non-retryable error classification
   - hard quota detection
   - circuit-breaker trip/open/close transitions
   - provider disablement after repeated rate-limit failures
   - mock response behavior by prompt version and test type
2. Add orchestrator tests for:
   - prompt-level progress callback invocation
   - ETA formatting and final run summary output
   - fallback-to-mock behavior when live execution fails completely
   - delta handling when one prompt version is missing data
3. Add regression tests for any bug that required a runtime patch.

**Verification:**
- `uv run pytest`
- `uv run python -m pytest tests -q`
- target the specific edited modules when making runner changes

**Exit criteria:**
- Tests exist for the live path, fallback path, and summary path.
- Regressions in provider throttling or progress output fail fast in CI.

**Dependency notes:** None. This can start immediately.

### Workstream 2: CI / Quality Gates
**Goal:** Ensure every push validates formatting, tests, and basic runtime health.

**Why this is next:** The repo is currently manual-run heavy. CI is the cheapest way to catch breakage before it reaches a demo or a review.

**Scope:**
1. Add a GitHub Actions workflow that runs on push and pull request.
2. Include at minimum:
   - dependency install with `uv`
   - lint/format check if the repo has a formatter configured
   - pytest execution
   - a quick smoke run of `main.py status`
3. Add a lightweight badge to `README.md` once the workflow is stable.
4. Prefer fast, deterministic checks over long live-provider runs in CI.

**Verification:**
- workflow completes on a clean checkout
- no secrets are required for the default CI path
- tests fail the workflow when regressions are introduced

**Exit criteria:**
- A fresh clone can self-check without manual steps.
- CI gives a clear pass/fail signal for core repo health.

**Dependency notes:** Depends on Workstream 1 for meaningful signal.

### Workstream 3: Reproducibility and Run Manifest
**Goal:** Make every results artifact defensible by capturing exactly what ran.

**Why this matters:** The current `results.json` is useful, but for reviews or interviews you need provenance: config, seed, provider state, and environment context.

**Scope:**
1. Save a run manifest alongside each experiment containing:
   - config snapshot
   - dataset target and generation parameters
   - active model list
   - enabled/disabled provider state
   - environment details relevant to the run
2. Capture a deterministic seed for dataset generation and record it in the artifact.
3. Persist a compact metadata block into SQLite for each run.
4. Add a `run_id`-indexed link between the summary JSON, CSV, SQLite rows, and manifest.
5. Ensure missing prompt versions or disabled providers are recorded as explicit absence, not hidden zeros.

**Verification:**
- rerunning with the same seed produces comparable data shapes
- manifest and artifact IDs match across JSON, CSV, and DB rows
- disabled-provider state is reflected in the run record

**Exit criteria:**
- A reviewer can reconstruct what ran without reading code.

**Dependency notes:** Depends on Workstream 2 if you want the manifest to be validated in CI.

### Workstream 4: Final Report Layer
**Goal:** Generate a concise human-readable report from the raw experiment artifacts.

**Why this is valuable:** JSON is good for machines; a readable report is what you show people.

**Scope:**
1. Add a report generator that summarizes:
   - baseline vs improved prompt metrics
   - top failure buckets
   - provider health / disablement events
   - runtime and average latency
   - live vs fallback execution counts
2. Produce both terminal output and a markdown or text artifact.
3. Include a short interpretation section that explains what changed and what failed.
4. Keep the report grounded in the live run only; do not blend mock fallback into the headline story unless it is explicitly labeled.

**Verification:**
- report can be generated from an existing `results.json`
- report matches the persisted JSON totals
- report is readable without opening the dashboard

**Exit criteria:**
- One command turns raw artifacts into an interview-ready summary.

**Dependency notes:** Depends on Workstream 3 for consistent artifact linkage.

### Workstream 5: Provider Health Policy and Re-enable Strategy
**Goal:** Make provider behavior adaptive instead of permanently pessimistic after a transient problem.

**Why this is needed:** The repo currently avoids rate limits by sidelining exhausted providers. That is safe, but it can be too sticky if a provider recovers later.

**Scope:**
1. Define provider health states such as:
   - healthy
   - throttled
   - disabled
   - cooling down
2. Add a policy for re-enabling providers after a quiet period or after the next run.
3. Distinguish hard quota exhaustion from transient rate limiting more explicitly.
4. Avoid re-enabling providers during the same run once a hard quota error is detected.
5. Keep the current conservative default behavior as the safe baseline.

**Verification:**
- a hard quota error disables the provider for the run
- transient rate-limit bursts can cool down and be retried in later runs
- provider state is visible in logs and/or the final report

**Exit criteria:**
- Provider decisions are explicit, documented, and observable.

**Dependency notes:** Depends on Workstream 1 and should feed Workstream 4.

### Workstream 6: Dashboard Polish and Operational View
**Goal:** Turn the Streamlit dashboard into a clear operational view rather than just a results viewer.

**Why this is later in the plan:** The dashboard is only as good as the data and summaries behind it, so the earlier workstream outputs should stabilize first.

**Scope:**
1. Add an executive summary panel:
   - success rate
   - elapsed time
   - average latency
   - top failure buckets
   - active providers
2. Add a provider health section showing:
   - enabled/disabled state
   - rate-limit history
   - last observed error
3. Add a prompt comparison panel that highlights baseline vs improved deltas clearly.
4. Make sure the dashboard degrades gracefully if some data is missing because a provider was disabled.
5. Keep the UI simple and legible rather than overloaded.

**Verification:**
- dashboard loads from the latest artifacts
- dashboard handles partial runs and missing models without crashing
- key metrics match the JSON summary

**Exit criteria:**
- The dashboard can explain the run without requiring the terminal output.

**Dependency notes:** Depends on Workstreams 3 and 4.

### Workstream 7: Repo Cleanup and Boundary Simplification
**Goal:** Remove leftover friction, reduce duplication, and make the main path easier to maintain.

**Why this is last:** Cleanup is most effective after the behavior is already stable and the reporting surface is settled.

**Scope:**
1. Separate legacy compatibility adapters from the main runtime path as much as possible.
2. Remove or isolate dead code, duplicate logic, and stale provider assumptions.
3. Tighten module boundaries between data generation, model execution, evaluation, and reporting.
4. Update README quickstart and operational notes so they match the actual final workflow.
5. Keep all changes minimal and behavior-preserving unless the cleanup intentionally replaces a brittle path.

**Verification:**
- main pipeline still runs after cleanup
- tests still pass
- README matches current commands and runtime behavior

**Exit criteria:**
- The repo is easier to navigate and explain in one pass.

**Dependency notes:** Depends on the earlier workstreams because it should preserve the final behavior, not redesign it.

## Recommended Order of Execution
1. Workstream 1
2. Workstream 2
3. Workstream 3
4. Workstream 4
5. Workstream 5
6. Workstream 6
7. Workstream 7

## Parallelism Notes
1. Workstreams 1 and 2 can overlap once the first test cases are in place.
2. Workstreams 3 and 5 can overlap after the manifest shape is agreed.
3. Workstreams 4 and 6 can overlap if they consume the same artifact contract.
4. Workstream 7 should wait until the others stabilize.

## Change Control Rules
1. Keep the live pipeline as the source of truth.
2. Avoid broad refactors while provider behavior is still changing.
3. Update the plan if a workstream reveals a hidden dependency.
4. If provider quotas change again, revisit Workstream 5 before expanding live validation size.
