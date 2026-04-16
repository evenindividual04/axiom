# Troubleshooting & FAQ

Common issues and solutions for Axiom.

## Installation & Setup

### "command not found: uv"

**Cause**: UV package manager not installed or not in PATH.

**Solution**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell
source $HOME/.cargo/env

# Verify
uv --version
```

### "Poetry lock or venv errors during bootstrap"

**Cause**: Python 3.10+ not available, or venv creation failed.

**Solution**:
```bash
# Check Python version
python3 --version     # Should be 3.10+

# Delete venv and retry
rm -rf .venv
bash scripts/bootstrap.sh

# Or manually create venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### ".env file not found"

**Cause**: Missing `.env` file with API keys.

**Solution**:
```bash
# Copy template
cp .env.example .env

# Edit file with your keys
vim .env
```

### "API key not found" error when running pipeline

**Cause**: Environment variable not loaded or API key invalid.

**Solution**:
```bash
# Verify .env exists in project root
ls -la .env

# Check key is set
echo $OPENAI_API_KEY    # Should print your key

# If empty, reload environment
source .env
uv run python main.py status

# Or verify in Python
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OPENAI_API_KEY:', os.getenv('OPENAI_API_KEY')[:20]...)"
```

---

## Configuration

### "Config file not found"

**Cause**: Wrong path or filename.

**Solution**:
```bash
# Verify config exists
ls -la config.yaml

# Use explicit path
uv run python main.py run-pipeline --config config.yaml
```

### "Unknown provider in config"

**Cause**: Typo in provider name (e.g., `openai_` instead of `openai`).

**Solution**:
Check `config.yaml` for typos. Valid providers:
- `openai`
- `anthropic`
- `groq`
- `gemini`
- `openrouter`
- `ollama`
- `zai`

```yaml
models:
  - name: gpt-4o-mini
    provider: openai          # ✅ Correct
    # provider: openai_      # ❌ Wrong
```

### "All models disabled in config"

**Cause**: No models have `enabled: true`.

**Solution**:
```yaml
# At least one model must be enabled
models:
  - name: gpt-4o-mini
    provider: openai
    enabled: true             # ✅ Change to true
```

---

## API & Authentication

### "Provider cooling down" or "Rate limited"

**Cause**: Provider rejected requests due to rate limit. Normal; Axiom handles this.

**Explanation**: When a provider returns 429 (Too Many Requests), Axiom:
1. Records the error
2. Triggers **circuit breaker** if threshold exceeded
3. Provider enters **cooling_down** state
4. Auto-recovers after cooldown expires (e.g., 120s)

**What to do**:
- Check dashboard **Provider Health** tab to see countdown
- Wait for cooldown to expire
- If persistent, reduce `concurrency` in config:

```yaml
runtime:
  concurrency: 1            # Reduce from 3
```

### "Provider disabled"

**Cause**: Hard failure (401 Unauthorized, 403 Forbidden, quota exhausted).

**Solution**:
1. Check your API account (quota, permissions, key validity)
2. Fix the issue (add quota, rotate key, check permissions)
3. Re-enable in config.yaml:

```yaml
models:
  - name: gpt-4o-mini
    provider: openai
    enabled: true            # Re-enable after fix
```

Or in the dashboard, request will auto-recover once provider is healthy.

### "OpenAI: 401 Unauthorized"

**Cause**: Invalid API key.

**Solution**:
```bash
# Verify key format (starts with sk-)
echo $OPENAI_API_KEY

# Check it's not empty
[ -z "$OPENAI_API_KEY" ] && echo "Key is empty"

# Test directly with curl
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### "Anthropic: 403 Forbidden"

**Cause**: API key invalid or account quota exceeded.

**Solution**:
```bash
# Test Anthropic API directly
curl https://api.anthropic.com/v1/models \
  -H "x-api-key: $ANTHROPIC_API_KEY"

# If 403, check:
# 1. API key is valid (console.anthropic.com)
# 2. Quota not exceeded
# 3. Billing active
```

### "Groq: Connection refused"

**Cause**: Groq API endpoint unreachable or wrong base URL.

**Solution**:
```bash
# Verify Groq endpoint
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"

# If timeout, may be outage; try later
```

### "Ollama: Connection refused"

**Cause**: Lokll Ollama server not running.

**Solution**:
```bash
# Start Ollama server
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags

# If not installed, install Ollama first
# https://ollama.ai/download
```

---

## Runtime & Execution

### "Dataset generation failed"

**Cause**: No documents in `data/fin-docs/` or documents are corrupted.

**Solution**:
```bash
# Check documents exist
ls -la data/fin-docs/
# Should see .txt, .md, or .pdf files

# If empty, download seed documents
uv run python scripts/download_seed_docs.py

# Or copy your own documents
cp /path/to/your/documents/* data/fin-docs/
```

### "Mock mode keeps running instead of live"

**Cause**: `runtime.mock_mode: true` in config.yaml.

**Solution**:
```yaml
runtime:
  mock_mode: false           # ✅ Set to false for live mode

# Or use command-line flag
uv run python main.py run-pipeline --live
```

### "Pipeline hangs or never finishes"

**Cause**: 
- Network timeout (slow provider)
- Provider not responding
- Infinite retry loop

**Solution**:
```bash
# Check if hung process
ps aux | grep "python main.py"

# Kill hung process
pkill -f "python main.py"

# Reduce timeout in config
runtime:
  request_timeout_seconds: 60     # Was 120; reduce if too long

# Reduce concurrency to lower load
runtime:
  concurrency: 1
```

### "Memory usage growing uncontrollably"

**Cause**: Large dataset + high concurrency holding too many async tasks in memory.

**Solution**:
```yaml
runtime:
  concurrency: 1                  # Reduce parallelism
  
dataset:
  target_rows: 50                 # Start smaller
  max_rows: 200                   # Cap size
```

### "Process crashed with SIGKILL"

**Cause**: Out of memory (OOM).

**Solution**:
```bash
# Reduce scope
uv run python main.py run-pipeline --live \
  --max-rows 50
```

---

## Data & Artifacts

### "results.json is empty or missing"

**Cause**: Pipeline failed before artifact emission or run hasn't completed.

**Solution**:
```bash
# Check if pipeline is still running
ps aux | grep "python main.py"

# Or check for errors in database
sqlite3 experiments/runs.db
> SELECT * FROM runs ORDER BY created_at DESC LIMIT 1;

# If no recent run, run pipeline again
uv run python main.py run-pipeline --live
```

### "row_results.csv is missing or truncated"

**Cause**: 
- Pipeline interrupted before CSV export
- All rows failed evaluation

**Solution**:
```bash
# Check SQLite for eval results
sqlite3 experiments/runs.db
> SELECT COUNT(*) FROM eval_rows WHERE run_id = 'run_...';

# If rows exist, manually export
python3 -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('experiments/runs.db')
df = pd.read_sql('SELECT * FROM eval_rows', conn)
df.to_csv('experiments/row_results.csv', index=False)
"
```

### "Manifest file corrupted or invalid JSON"

**Cause**: Write interrupted or disk full.

**Solution**:
```bash
# Validate JSON
python3 -c "import json; json.load(open('experiments/manifests/run_*.json'))"

# If corrupted, delete and rerun pipeline
rm experiments/manifests/run_*.json
uv run python main.py run-pipeline --live
```

### "Can't find latest report symlink"

**Cause**: Latest symlink is stale or pointing to deleted file.

**Solution**:
```bash
# Recreate latest symlink
cd experiments/reports
ln -sf run_*.md latest.md

# Or just view most recent directly
ls -lt run_*.md | head -1 | awk '{print $NF}' | xargs cat
```

---

## Dashboard

### "Dashboard won't start"

**Cause**: Streamlit error, port conflict, or missing results.

**Solution**:
```bash
# Check results.json exists
ls -la experiments/results.json

# If not, run pipeline first
uv run python main.py run-pipeline --live

# Clean cache and restart
rm -rf ~/.streamlit/cache
pkill -f "streamlit run"
uv run python main.py dashboard
```

### "Dashboard shows "ERROR: No run results found""

**Cause**: `experiments/results.json` missing or pipeline failed.

**Solution**:
```bash
# Run pipeline to generate artifacts
uv run python main.py run-pipeline --live

# Verify results.json was created
cat experiments/results.json | jq .
```

### "Provider Health tab is empty"

**Cause**: Manifest doesn't include `provider_health_snapshot`.

**Solution**:
```bash
# Check manifest contains health data
cat experiments/manifests/run_*.json | jq '.provider_health_snapshot'

# If missing, rerun pipeline (older code may not have it)
uv run python main.py run-pipeline --live
```

### "Dashboard doesn't refresh when new artifacts are added"

**Cause**: Streamlit cache is stale.

**Solution**:
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache
rm -rf .streamlit/cache

# Restart dashboard
pkill -f "streamlit run"
uv run python main.py dashboard
```

### "Plots/charts not showing in dashboard"

**Cause**: Missing plotly or pandas, or data is empty.

**Solution**:
```bash
# Check dependencies installed
uv pip list | grep -E 'plotly|pandas'

# Or reinstall
bash scripts/bootstrap.sh

# Verify data is not empty
sqlite3 experiments/runs.db "SELECT COUNT(*) FROM eval_rows;"
```

---

## Testing & Validation

### "Tests fail: "ModuleNotFoundError""

**Cause**: Project dependencies not installed or Python path wrong.

**Solution**:
```bash
# Install dev dependencies
uv sync --extra dev

# Or reinstall
bash scripts/bootstrap.sh

# Then run tests
pytest -v
```

### "Tests fail: "DatabaseError" or "no such table""

**Cause**: SQLite schema not initialized during test.

**Solution**:
```bash
# Check conftest.py sets up schema
cat tests/conftest.py | grep -A 10 "sqlite"

# Run single test in verbose mode
pytest -v tests/test_orchestrator_manifest.py::test_name -s
```

### "Coverage report showing 0%"

**Cause**: Coverage data not collected; pytest-cov not configured.

**Solution**:
```bash
# Run pytest with coverage
pytest --cov=src/llm_reliability_engine --cov=evals --cov=models

# Or check pyproject.toml has pytest config
cat pyproject.toml | grep -A 5 "tool.pytest"
```

---

## Performance Tuning

### "Evals taking too long"

**Cause**: Sequential execution or low concurrency.

**Solution**:
```yaml
runtime:
  concurrency: 5                  # Increase from 3 (if rates allow)
```

Or profile the bottleneck:
```bash
# Time a single model call
time uv run python -c "
from models.llm_runner import run_models
import asyncio
# ... measure latency
"
```

### "Provider rate limiting too aggressive"

**Cause**: Concurrency too high or global rate limit exceeded.

**Solution**:
```yaml
runtime:
  concurrency: 1                  # Lower parallelism
  rate_limit_per_minute: 3        # More conservative

  provider_overrides:
    openai:
      rate_limit_per_minute: 5
```

### "Dashboard queries are slow"

**Cause**: SQLite not indexed or too many rows.

**Solution**:
```bash
# Analyze database
sqlite3 experiments/runs.db
> ANALYZE;
> PRAGMA table_info(eval_rows);

# If many rows, consider partitioning
# Or upgrade to PostgreSQL (see ARCHITECTURE.md)
```

---

## Common Failures & Solutions

| Symptom | Cause | Fix |
|---------|-------|-----|
| "All models error on first request" | API keys invalid or all providers down | Check .env, verify API account status |
| "Specific model always timeout" | Model slow, timeout too short | Increase `request_timeout_seconds` |
| "Hallucination scores unreasonably high" | Grounding documents too small/irrelevant | Add more domain documents to `data/fin-docs/` |
| "Success rate is 0%" | Test cases impossible or prompts broken | Review ground_truth in manifest; test prompt manually |
| "Provider keeps cycling cooling_down" | Rate limit too aggressive; config too strict | Lower rate_limit_per_minute or reduce concurrency |
| "SQLite locked" | Multiple processes writing simultaneously | Ensure only one `main.py run-pipeline` at a time |

---

## Getting More Help

1. **Check logs**: `cat experiments/runs.db` or check console output during pipeline run
2. **Inspect artifacts**: `cat experiments/manifests/run_*.json | jq` to see full details
3. **Review tests**: `pytest -v -s` to see what's passing/failing
4. **Check docs**: `docs/ARCHITECTURE.md` has detailed component reference
5. **Debug interactively**: `python3 -i` and import modules to test manually
6. **GitHub issues**: File an issue with config, error message, and steps to reproduce

---

## Tips for Debugging

### Print full error stack trace

```python
# In Python code
import traceback
try:
    # ... code
except Exception as e:
    traceback.print_exc()
```

### Enable asyncio debugging

```python
# In models/llm_runner.py
import asyncio
asyncio.run(main(), debug=True)
```

### Inspect SQLite database directly

```bash
# Open interactive shell
sqlite3 experiments/runs.db

# Useful queries
.schema                                    # Show all tables
SELECT COUNT(*) FROM eval_rows;            # Total rows evaluated
SELECT DISTINCT failure_type FROM eval_rows;  # Failure types encountered
SELECT model, COUNT(*) FROM eval_rows GROUP BY model;  # Results per model
```

### Test a single provider in isolation

```python
# Test OpenAI directly
import asyncio
from models.llm_runner import _call_openai, ProviderRuntimeSettings

output = asyncio.run(_call_openai(
    model="gpt-4o-mini",
    prompt="base",
    question="What is 2+2?",
    context="Math facts",
    ground_truth="4",
    api_key=os.getenv("OPENAI_API_KEY"),
    settings=ProviderRuntimeSettings(
        rate_limit_per_minute=6,
        max_retries=2,
        # ...
    )
))
print(output)
```

---

## Reporting Bugs

If you encounter an issue:

1. **Try the solutions above first**
2. **Reproduce with minimal config**:
   ```bash
   # Simplest setup: 1 model, 10 rows, no mock fallback
   # config.yaml:
   dataset:
     target_rows: 10
   models:
     - name: gpt-4o-mini
       provider: openai
       enabled: true
   runtime:
     mock_mode: false
     mock_fallback_on_failure: false
   ```
3. **Collect diagnostics**:
   ```bash
   uv run python main.py status
   cat experiments/manifests/*.json | jq .runtime  # Check config snapshot
   sqlite3 experiments/runs.db "SELECT * FROM eval_rows LIMIT 1;"
   ```
4. **File issue** with:
   - Error message (full stack trace)
   - Config (without sensitive keys)
   - Steps to reproduce
   - Platform (macOS/Linux/Windows) and Python version
