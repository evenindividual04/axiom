# Getting Started with Axiom

This guide walks you through setting up Axiom for your first evaluation run.

## 1. Clone and Install

```bash
git clone https://github.com/evenindividual04/axiom.git
cd axiom

# Install dependencies and create environment
bash scripts/bootstrap.sh

# Verify installation
uv run python main.py status
```

Expected output:
```json
{
  "project": "axiom",
  "status": "ready",
  "timestamp": "2026-04-17T14:30:00+00:00"
}
```

## 2. Prepare API Keys

Create `.env` file in project root:

```bash
cp .env.example .env
```

Edit `.env` and add your provider API keys (only providers you want to test):

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Groq
GROQ_API_KEY=gsk-...

# Gemini
GEMINI_API_KEY=...

# OpenRouter
OPENROUTER_API_KEY=sk-or-...

# ZAI (if using Chinese models)
ZAI_API_KEY=...

# Optional: Custom base URLs
OPENAI_BASE_URL=https://api.openai.com/v1
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

For **Ollama** models, ensure the Ollama server is running locally:
```bash
ollama serve
```

## 3. Enable Models in Config

Edit `config.yaml` and enable the models you want to test:

```yaml
models:
  - name: gpt-4o-mini
    provider: openai
    enabled: true              # Enable
  - name: claude-3-5-haiku-latest
    provider: anthropic
    enabled: true              # Enable
  - name: gemini-2.0-flash
    provider: gemini
    enabled: false             # Skip
```

Supported providers:
- `openai`: GPT-4o, GPT-4o-mini, etc.
- `anthropic`: Claude 3.5, Claude 3, etc.
- `groq`: Llama 3.3 70B, Mixtral, etc.
- `gemini`: Gemini Flash, Gemini Pro, etc.
- `openrouter`: Aggregate + custom routing
- `ollama`: Local open-source models
- `zai`: Chinese models (GLM, etc.)

## 4. Customize Data (Critical)

Axiom generates test cases from your domain documents. This is what makes it domain-agnostic.

### Option A: Use Seed Documents (Default)

The project ships with financial documents as examples. To keep them:
```bash
uv run python scripts/download_seed_docs.py
```

This pulls documents from URLs in `scripts/seed_doc_sources.json`.

### Option B: Add Your Domain Documents

Place documents in `data/fin-docs/` (supported: `.txt`, `.md`, `.pdf`):

```bash
# Example: Healthcare
cp /path/to/clinical_guidelines.pdf data/fin-docs/
cp /path/to/treatment_protocols.txt data/fin-docs/

# Example: Legal
cp /path/to/case_law.md data/fin-docs/

# Example: E-commerce
cp /path/to/product_catalog.txt data/fin-docs/
```

### Option C: Replace Seed Sources Entirely

Edit `scripts/seed_doc_sources.json`:
```json
{
  "sources": [
    "https://your-domain-experts.org/knowledge-base/",
    "https://your-company.com/documentation/",
    "https://public-database.gov/records/"
  ],
  "source_description": "Your domain knowledge base"
}
```

Then:
```bash
uv run python scripts/download_seed_docs.py
```

## 5. Customize Test Prompts (Critical)

Axiom compares prompt versions. By default, it ships with financial prompts. **Customize these for your domain:**

### Edit `prompts/base_prompt.txt`

Replace the financial prompt with your domain question:

**Healthcare example:**
```
You are a clinical decision support system. Based on the provided medical literature and guidelines, 
answer the following question accurately and comprehensively:

{question}

Grounding:
{context}

Ground Truth:
{ground_truth}

Provide your analysis in 2-3 sentences, focusing on clinical safety and evidence-based practice.
```

**Legal example:**
```
You are a legal research assistant. Based on the provided case law and statutes, 
answer the following legal question:

{question}

Relevant law:
{context}

Expected answer:
{ground_truth}

Cite specific precedents in your response.
```

**E-commerce example:**
```
You are a customer support AI. Based on the product knowledge base and policies, 
answer this customer question:

{question}

Product info:
{context}

Company answer:
{ground_truth}

Be accurate, helpful, and concise.
```

### Create Variants

Create `prompts/improved_prompt.txt` and `prompts/advanced_prompt.txt` with refinements:
- Clearer instructions
- Better examples
- Stricter output format requirements
- Domain-specific guardrails

Axiom will compare all prompt versions and show which one improves your metrics.

## 6. Run Your First Eval

```bash
uv run python main.py run-pipeline --live
```

This will:
1. ✅ Load your domain documents
2. ✅ Generate 100 synthetic test cases (configurable)
3. ✅ Run all enabled models in parallel
4. ✅ Compute metrics per row (relevancy, faithfulness, hallucination, safety)
5. ✅ Classify failures
6. ✅ Track provider health (circuit breaker, rate limits)
7. ✅ Emit manifest, report, CSV artifacts

Typical runtime for 100 rows × 3 models: **2–5 minutes** (depending on parallelism and provider latency).

**Output:**
```
{
  "run_id": "run_2026-04-17_12-34-56_abc123",
  "total_rows": 100,
  "evaluated_rows": 98,
  "failed_rows": 2,
  "manifest_path": "experiments/manifests/run_2026-04-17_12-34-56_abc123.json",
  "report_path": "experiments/reports/run_2026-04-17_12-34-56_abc123.md",
  "csv_path": "experiments/row_results.csv",
  "results_path": "experiments/results.json"
}
```

## 7. Explore Results

### Via Dashboard (Recommended)

```bash
uv run python main.py dashboard
```

Opens Streamlit UI at `http://localhost:8501` with:
- **Overview**: Success rate, failure counts, average metrics
- **Prompts**: Baseline vs improved prompt deltas
- **Provider Health**: Current state of each provider (healthy/cooling_down/disabled)
- **Artifacts**: Links to full manifest, report, CSV

### Via CSV (Raw Data)

```bash
# All row-level results
cat experiments/row_results.csv | head -20

# Columns: model, prompt_version, question, output, 
#          answer_relevancy, faithfulness, hallucination_rate, safety_score, 
#          failure_type, latency_ms, token_count
```

### Via Markdown Report (Human-Readable)

```bash
# Latest run report
cat experiments/reports/latest.md
```

Shows:
- Run metadata (when, which models, which prompts)
- Summary metrics (success rate, failure breakdown)
- Failure examples with classification
- Prompt comparison deltas
- Provider health snapshot

### Via Manifest (Full Provenance)

```bash
# Full evaluation record with all inputs/outputs
cat experiments/manifests/run_*.json | jq .
```

Includes:
- Every test case (question, context, ground truth)
- Every model response (output, latency, token count)
- Every metric computed (relevancy, faithfulness, etc.)
- Provider health state at evaluation time
- Config snapshot

## 8. Next Steps

### Iterate on Prompts

1. Edit `prompts/improved_prompt.txt`
2. Run eval: `uv run python main.py run-pipeline --live`
3. Check dashboard to see which prompt wins

### Add More Models

Edit `config.yaml` and enable more providers:
```yaml
models:
  - name: gpt-4-turbo
    provider: openai
    enabled: true
  - name: claude-3-opus-latest
    provider: anthropic
    enabled: true
  - name: llama-3.3-70b-specdec
    provider: groq
    enabled: true
```

### Tune Configuration

Adjust `runtime` settings in `config.yaml`:
```yaml
runtime:
  concurrency: 5                    # More parallel requests
  request_timeout_seconds: 120      # Longer timeout
  circuit_breaker_cooldown_seconds: 180  # Slower recovery
```

### Add Custom Metrics

Edit `evals/metrics.py` to add your domain-specific metrics:
```python
def domain_specific_metric(output: str, context: str) -> float:
    # Your evaluation logic
    return score
```

Then integrate in `evals/pipeline.py`:
```python
custom_score = domain_specific_metric(output, context)
```

### Run Tests

```bash
pytest -v
pytest --cov=src/llm_reliability_engine --cov=evals --cov=models
```

## Troubleshooting

### "API key not found" or "unauthorized"

```bash
# Verify .env is in project root
ls -la .env

# Check key is set
printenv OPENAI_API_KEY

# Reload environment
source .env
uv run python main.py status
```

### "Provider cooling down" or "disabled"

Check dashboard **Provider Health** tab. Common causes:
- Rate limit hit: Wait for cooldown (default 120s)
- Quota exhausted: Check API account, increase quota
- Temporary outage: Provider health will auto-recover

Manual fix:
```yaml
# Temporarily disable troublesome provider in config.yaml
- name: problematic-model
  provider: groq
  enabled: false
```

### "Dataset generation failed"

Ensure documents are in place:
```bash
ls -la data/fin-docs/
# Should see your .txt, .md, or .pdf files

# Fallback: generate empty synthetic data
uv run python main.py run-pipeline --live --skip-grounding
```

### "Dashboard not loading"

```bash
# Check results exist
ls -la experiments/results.json

# Restart dashboard
Ctrl+C
uv run python main.py dashboard
```

If still blank, run pipeline first:
```bash
uv run python main.py run-pipeline --live
```

## Tips for Success

1. **Start with 1–2 models**: Test with a fast, cheap model first (e.g., `gpt-4o-mini`). Add more models later.

2. **Use mock mode for quick iteration**:
   ```bash
   # Fast mock evaluation (no API costs)
   uv run python main.py run-pipeline
   ```

3. **Test your documents**: Ensure `data/fin-docs/` has reasonable domain content. Axiom uses this to gen test cases.

4. **Version your prompts**: Keep old versions in `prompts/` directory. Dashboard lets you compare.

5. **Set alerts on failures**: Check the markdown report for high failure counts on specific models. That signals a capability gap.

## Getting Help

- Check `README.md` for overview
- See `docs/ARCHITECTURE.md` for technical deep dive
- See `docs/DOMAIN_SETUP.md` for domain-specific setup examples
- Check existing issues: https://github.com/evenindividual04/axiom/issues
- Run tests: `pytest -v` to validate setup
