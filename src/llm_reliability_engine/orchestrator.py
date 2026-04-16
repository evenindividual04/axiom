from __future__ import annotations

import asyncio
import csv
import json
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable

import yaml

from data.synthetic.generate_dataset import generate_dataset
from evals.pipeline import EvalResult, evaluate_row, summarize, to_dict
from models.llm_runner import ModelError, ProviderRuntimeSettings, run_models
from llm_reliability_engine.reporting import build_run_report_markdown, write_run_report

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
MANIFESTS_DIR = EXPERIMENTS_DIR / "manifests"
REPORTS_DIR = EXPERIMENTS_DIR / "reports"
PROMPTS_DIR = ROOT / "prompts"
DB_PATH = EXPERIMENTS_DIR / "runs.db"
RESULTS_PATH = EXPERIMENTS_DIR / "results.json"
ROWS_PATH = EXPERIMENTS_DIR / "row_results.csv"


def _load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _read_prompt(version: str) -> str:
    prompt_file = PROMPTS_DIR / f"{version}_prompt.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file missing: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            mock_mode INTEGER NOT NULL,
            target_rows INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS eval_rows (
            run_id TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            question TEXT NOT NULL,
            output TEXT NOT NULL,
            answer_relevancy REAL NOT NULL,
            faithfulness REAL NOT NULL,
            hallucination_rate REAL NOT NULL,
            safety_score REAL NOT NULL,
            failure_type TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            token_count INTEGER NOT NULL
        );
        """
    )
    conn.commit()


def _persist_run(conn: sqlite3.Connection, run_id: str, mock_mode: bool, target_rows: int) -> None:
    conn.execute(
        "INSERT INTO runs(run_id, created_at, mock_mode, target_rows) VALUES(?,?,?,?)",
        (run_id, datetime.now(UTC).isoformat(), int(mock_mode), target_rows),
    )
    conn.commit()


def _persist_rows(conn: sqlite3.Connection, run_id: str, rows: list[EvalResult]) -> None:
    payload = [
        (
            run_id,
            row.model,
            row.prompt_version,
            row.question,
            row.output,
            row.answer_relevancy,
            row.faithfulness,
            row.hallucination_rate,
            row.safety_score,
            row.failure_type,
            row.latency_ms,
            row.token_count,
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO eval_rows(
            run_id, model, prompt_version, question, output,
            answer_relevancy, faithfulness, hallucination_rate, safety_score,
            failure_type, latency_ms, token_count
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        payload,
    )
    conn.commit()


def _write_rows_csv(rows: list[EvalResult]) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with ROWS_PATH.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "model",
                "prompt_version",
                "question",
                "output",
                "answer_relevancy",
                "faithfulness",
                "hallucination_rate",
                "safety_score",
                "failure_type",
                "latency_ms",
                "token_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(to_dict(row))


def _group_summary(rows: list[EvalResult]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[EvalResult]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row.model][row.prompt_version].append(row)

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, prompt_map in grouped.items():
        summary[model_name] = {}
        for prompt_version, values in prompt_map.items():
            summary[model_name][prompt_version] = summarize(values)
    return summary


def _baseline_improved(summary: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_name, prompt_map in summary.items():
        baseline = prompt_map.get("base", {})
        improved = prompt_map.get("improved", {})

        baseline_has_data = bool(baseline) and baseline.get("count", 0) > 0
        improved_has_data = bool(improved) and improved.get("count", 0) > 0

        if baseline_has_data and improved_has_data:
            delta = {
                "hallucination_rate": round(improved.get("hallucination_rate", 0.0) - baseline.get("hallucination_rate", 0.0), 4),
                "faithfulness": round(improved.get("faithfulness", 0.0) - baseline.get("faithfulness", 0.0), 4),
                "safety_score": round(improved.get("safety_score", 0.0) - baseline.get("safety_score", 0.0), 4),
            }
        else:
            delta = {
                "hallucination_rate": None,
                "faithfulness": None,
                "safety_score": None,
            }

        result[model_name] = {
            "baseline": baseline,
            "improved": improved,
            "delta": delta,
        }
    return result


@dataclass(frozen=True)
class ProviderHealthState:
    status: str = "healthy"
    reason: str = ""
    disabled_until: float | None = None
    last_error: str = ""
    transient_error_count: int = 0


class ProviderHealthPolicy:
    def __init__(self, provider_settings: dict[str, ProviderRuntimeSettings] | None = None):
        self._provider_settings = dict(provider_settings or {})
        self._states: dict[str, ProviderHealthState] = {}

    def _cooldown_seconds(self, provider: str) -> float:
        settings = self._provider_settings.get(provider)
        if settings is not None:
            return max(1.0, settings.circuit_breaker_cooldown_seconds)
        return 300.0

    def _state_for(self, provider: str) -> ProviderHealthState:
        return self._states.get(provider, ProviderHealthState())

    def _store_state(self, provider: str, state: ProviderHealthState) -> None:
        self._states[provider] = state

    def _refresh_expired(self, provider: str, now: float) -> ProviderHealthState:
        state = self._state_for(provider)
        if state.status == "cooling_down" and state.disabled_until is not None and now >= state.disabled_until:
            refreshed = ProviderHealthState(status="healthy")
            self._store_state(provider, refreshed)
            return refreshed
        return state

    def record_errors(self, errors: list[ModelError], now: float | None = None) -> None:
        current_time = time.monotonic() if now is None else now
        by_provider: dict[str, list[ModelError]] = defaultdict(list)
        for err in errors:
            provider = err.provider.strip().lower()
            if provider:
                by_provider[provider].append(err)

        for provider, provider_errors in by_provider.items():
            state = self._refresh_expired(provider, current_time)
            last_error = provider_errors[-1].error if provider_errors else state.last_error

            if any(_is_hard_quota_error(err.error) for err in provider_errors):
                self._store_state(
                    provider,
                    ProviderHealthState(
                        status="disabled",
                        reason="hard_quota",
                        disabled_until=None,
                        last_error=last_error,
                        transient_error_count=0,
                    ),
                )
                continue

            transient_count = sum(1 for err in provider_errors if _is_rate_limit_error(err.error))
            if transient_count >= 3:
                next_disabled_until = current_time + self._cooldown_seconds(provider)
                if state.status == "cooling_down" and state.disabled_until is not None:
                    next_disabled_until = max(next_disabled_until, state.disabled_until)
                self._store_state(
                    provider,
                    ProviderHealthState(
                        status="cooling_down",
                        reason="rate_limit",
                        disabled_until=next_disabled_until,
                        last_error=last_error,
                        transient_error_count=max(state.transient_error_count, transient_count),
                    ),
                )

    def apply(self, model_specs: list[dict[str, Any]], now: float | None = None) -> list[dict[str, Any]]:
        current_time = time.monotonic() if now is None else now
        updated_specs: list[dict[str, Any]] = []

        for spec in model_specs:
            provider = str(spec.get("provider", "")).strip().lower()
            base_enabled = bool(spec.get("enabled", True))
            updated_spec = dict(spec)
            state = self._refresh_expired(provider, current_time) if provider else ProviderHealthState()

            if not base_enabled:
                updated_spec["enabled"] = False
            elif state.status in {"disabled", "cooling_down"}:
                updated_spec["enabled"] = False
            else:
                updated_spec["enabled"] = True

            updated_specs.append(updated_spec)

        return updated_specs

    def snapshot(self, now: float | None = None) -> dict[str, dict[str, Any]]:
        current_time = time.monotonic() if now is None else now
        snapshot: dict[str, dict[str, Any]] = {}
        for provider, state in self._states.items():
            effective_state = self._refresh_expired(provider, current_time)
            snapshot[provider] = {
                "status": effective_state.status,
                "reason": effective_state.reason,
                "disabled_until": effective_state.disabled_until,
                "last_error": effective_state.last_error,
                "transient_error_count": effective_state.transient_error_count,
            }
        return snapshot


def _try_log_mlflow(run_id: str, summary: dict[str, dict[str, dict[str, Any]]]) -> None:
    try:
        import mlflow
    except Exception:
        return

    mlflow.set_experiment("llm-reliability-engine")
    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("run_id", run_id)
        for model_name, prompt_map in summary.items():
            for prompt_version, stats in prompt_map.items():
                prefix = f"{model_name}.{prompt_version}"
                for key in ("answer_relevancy", "faithfulness", "hallucination_rate", "safety_score", "latency_ms", "token_count"):
                    value = stats.get(key)
                    if isinstance(value, (float, int)):
                        mlflow.log_metric(f"{prefix}.{key}", float(value))


async def _evaluate_for_prompt(
    *,
    dataset_rows: list[dict[str, Any]],
    model_specs: list[dict[str, Any]],
    prompt_version: str,
    prompt_template: str,
    mock_mode: bool,
    concurrency: int,
    timeout_seconds: float,
    provider_runtime_settings: dict[str, ProviderRuntimeSettings],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[EvalResult], list[ModelError]]:
    model_outputs, errors = await run_models(
        dataset=dataset_rows,
        model_specs=model_specs,
        prompt_version=prompt_version,
        prompt_template=prompt_template,
        mock_mode=mock_mode,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        provider_runtime_settings=provider_runtime_settings,
        progress_callback=progress_callback,
    )

    results = [
        evaluate_row(
            model=output.model,
            prompt_version=output.prompt_version,
            question=output.question,
            context=output.context,
            ground_truth=output.ground_truth,
            output=output.output,
            latency_ms=output.latency_ms,
            token_count=output.token_count,
        )
        for output in model_outputs
    ]
    return results, errors


def _resolve_mock_mode(config: dict[str, Any], force_live: bool) -> bool:
    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or "mock_mode" not in runtime:
        raise ValueError("runtime.mock_mode must be explicitly set in config.yaml")

    raw = runtime["mock_mode"]
    if not isinstance(raw, bool):
        raise ValueError("runtime.mock_mode must be a boolean")

    mock_mode = raw
    if force_live:
        mock_mode = False
    return mock_mode


def _provider_runtime_settings(config: dict[str, Any], model_specs: list[dict[str, Any]]) -> dict[str, ProviderRuntimeSettings]:
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    provider_overrides_raw = runtime.get("provider_overrides", {}) if isinstance(runtime.get("provider_overrides"), dict) else {}
    provider_overrides = {
        str(key).strip().lower(): value
        for key, value in provider_overrides_raw.items()
    }

    default = ProviderRuntimeSettings(
        rate_limit_per_minute=int(runtime.get("rate_limit_per_minute", 60)),
        max_retries=int(runtime.get("max_retries", 2)),
        retry_base_delay_seconds=float(runtime.get("retry_base_delay_seconds", 0.5)),
        retry_max_delay_seconds=float(runtime.get("retry_max_delay_seconds", 8.0)),
        circuit_breaker_failure_threshold=int(runtime.get("circuit_breaker_failure_threshold", 5)),
        circuit_breaker_cooldown_seconds=float(runtime.get("circuit_breaker_cooldown_seconds", 30.0)),
    )

    settings: dict[str, ProviderRuntimeSettings] = {}
    for spec in model_specs:
        provider = str(spec.get("provider", "")).strip().lower()
        if not provider:
            continue
        if provider in settings:
            continue
        override = provider_overrides.get(provider, {}) if isinstance(provider_overrides.get(provider), dict) else {}
        settings[provider] = ProviderRuntimeSettings(
            rate_limit_per_minute=int(override.get("rate_limit_per_minute", default.rate_limit_per_minute)),
            max_retries=int(override.get("max_retries", default.max_retries)),
            retry_base_delay_seconds=float(override.get("retry_base_delay_seconds", default.retry_base_delay_seconds)),
            retry_max_delay_seconds=float(override.get("retry_max_delay_seconds", default.retry_max_delay_seconds)),
            circuit_breaker_failure_threshold=int(
                override.get("circuit_breaker_failure_threshold", default.circuit_breaker_failure_threshold)
            ),
            circuit_breaker_cooldown_seconds=float(
                override.get("circuit_breaker_cooldown_seconds", default.circuit_breaker_cooldown_seconds)
            ),
        )
    return settings


def _is_rate_limit_error(error_message: str) -> bool:
    message = error_message.lower()
    tokens = [
        "429",
        "rate limit",
        "too many requests",
        "quota exceeded",
        "resource_exhausted",
        "insufficient_quota",
    ]
    return any(token in message for token in tokens)


def _is_hard_quota_error(error_message: str) -> bool:
    message = error_message.lower()
    hard_quota_tokens = [
        "quota exceeded",
        "requests per day",
        "limit: 0",
        "free_tier_requests",
        "free_tier_input_token_count",
        "insufficient_quota",
        "billing details",
    ]
    return any(token in message for token in hard_quota_tokens)


def _disable_rate_limited_models(model_specs: list[dict[str, Any]], errors: list[ModelError]) -> list[dict[str, Any]]:
    policy = ProviderHealthPolicy()
    policy.record_errors(errors)
    return policy.apply(model_specs)


def _error_reason_bucket(error_message: str) -> str:
    message = error_message.lower()
    if "circuit breaker open" in message:
        return "circuit_breaker_open"
    if _is_hard_quota_error(error_message):
        return "hard_quota"
    if _is_rate_limit_error(error_message):
        return "rate_limit"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "connection" in message:
        return "connection"
    return "other"


def _print_prompt_summary(
    *,
    prompt_version: str,
    total_tasks: int,
    prompt_results: list[EvalResult],
    prompt_errors: list[ModelError],
    fallback_used: bool,
) -> None:
    success_count = len(prompt_results)
    error_count = len(prompt_errors)
    denominator = total_tasks if total_tasks > 0 else max(1, success_count + error_count)
    success_rate = (success_count / denominator) * 100

    reason_counts = Counter(_error_reason_bucket(err.error) for err in prompt_errors)
    top_reasons = ", ".join(
        f"{reason}:{count}" for reason, count in reason_counts.most_common(3)
    )
    if not top_reasons:
        top_reasons = "none"

    print(
        (
            f"[summary] prompt={prompt_version} total={denominator} success={success_count} "
            f"errors={error_count} success_rate={success_rate:.1f}% "
            f"fallback={str(fallback_used).lower()} top_errors={top_reasons}"
        ),
        flush=True,
    )


def _print_run_summary(*, run_started_at: float, results: list[EvalResult], errors: list[ModelError]) -> None:
    elapsed = max(0.0, time.perf_counter() - run_started_at)
    total_rows = len(results) + len(errors)
    average_latency = sum(row.latency_ms for row in results) / len(results) if results else 0.0
    success_rate = (len(results) / total_rows) * 100 if total_rows > 0 else 0.0

    print(
        (
            f"[run-summary] rows={total_rows} success={len(results)} errors={len(errors)} "
            f"success_rate={success_rate:.1f}% elapsed={elapsed:.1f}s avg_latency_ms={average_latency:.1f}"
        ),
        flush=True,
    )


def _path_from_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _enabled_providers(model_specs: list[dict[str, Any]]) -> list[str]:
    providers = {
        str(spec.get("provider", "")).strip().lower()
        for spec in model_specs
        if spec.get("enabled", True)
    }
    return sorted(provider for provider in providers if provider)


def _enabled_models(model_specs: list[dict[str, Any]]) -> list[str]:
    labels = {
        f"{str(spec.get('provider', '')).strip().lower()}/{str(spec.get('model', '')).strip()}"
        for spec in model_specs
        if spec.get("enabled", True)
    }
    return sorted(label for label in labels if "/" in label and not label.endswith("/"))


def _build_run_manifest(
    *,
    run_id: str,
    config_file: Path,
    config: dict[str, Any],
    force_live: bool,
    mock_mode: bool,
    mock_fallback_on_failure: bool,
    target_rows: int,
    dataset_path: Path,
    dataset_rows: list[dict[str, Any]],
    prompt_versions: list[str],
    initial_model_specs: list[dict[str, Any]],
    final_model_specs: list[dict[str, Any]],
    provider_settings: dict[str, ProviderRuntimeSettings],
    provider_health_snapshot: dict[str, dict[str, Any]],
    run_started_wall_clock: datetime,
    run_started_perf: float,
    results: list[EvalResult],
    errors: list[ModelError],
) -> dict[str, Any]:
    finished_at = datetime.now(UTC)
    total_rows = len(results) + len(errors)
    success_rate = (len(results) / total_rows) * 100 if total_rows > 0 else 0.0
    error_buckets = Counter(_error_reason_bucket(err.error) for err in errors)

    dataset_test_type_counts = Counter(str(row.get("test_type", "unknown")) for row in dataset_rows)
    dataset_domain_counts = Counter(str(row.get("domain", "unknown")) for row in dataset_rows)

    enabled_providers_start = _enabled_providers(initial_model_specs)
    enabled_providers_end = _enabled_providers(final_model_specs)
    disabled_providers = sorted(set(enabled_providers_start) - set(enabled_providers_end))

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at": run_started_wall_clock.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(max(0.0, time.perf_counter() - run_started_perf), 3),
        "config": {
            "path": _path_from_root(config_file),
            "snapshot": config,
        },
        "runtime": {
            "force_live": force_live,
            "mock_mode": mock_mode,
            "mock_fallback_on_failure": mock_fallback_on_failure,
            "target_rows": target_rows,
            "prompt_versions": prompt_versions,
        },
        "dataset": {
            "source_csv": _path_from_root(dataset_path),
            "generated_rows": len(dataset_rows),
            "test_type_counts": dict(dataset_test_type_counts),
            "domain_counts": dict(dataset_domain_counts),
        },
        "providers": {
            "runtime_settings": {
                provider: asdict(settings)
                for provider, settings in provider_settings.items()
            },
            "health_snapshot": provider_health_snapshot,
            "enabled_at_start": enabled_providers_start,
            "enabled_at_end": enabled_providers_end,
            "disabled_during_run": disabled_providers,
        },
        "models": {
            "enabled_at_start": _enabled_models(initial_model_specs),
            "enabled_at_end": _enabled_models(final_model_specs),
        },
        "outcomes": {
            "success_count": len(results),
            "error_count": len(errors),
            "success_rate": round(success_rate, 2),
            "error_reason_counts": dict(error_buckets),
        },
        "artifacts": {
            "results_json": _path_from_root(RESULTS_PATH),
            "row_results_csv": _path_from_root(ROWS_PATH),
            "runs_db": _path_from_root(DB_PATH),
        },
    }
    return manifest


def _write_run_manifest(manifest: dict[str, Any]) -> Path:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = str(manifest.get("run_id", "run_unknown"))
    manifest_path = MANIFESTS_DIR / f"{run_id}.manifest.json"
    latest_path = MANIFESTS_DIR / "latest.manifest.json"

    encoded = json.dumps(manifest, indent=2)
    manifest_path.write_text(encoded, encoding="utf-8")
    latest_path.write_text(encoded, encoding="utf-8")
    return manifest_path


def _new_run_id(now: datetime | None = None) -> str:
    instant = now or datetime.now(UTC)
    return f"run_{instant.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"


def run_pipeline(config_path: Path | None = None, force_live: bool = False) -> dict[str, Any]:
    config_file = config_path or (ROOT / "config.yaml")
    config = _load_config(config_file)
    run_id = _new_run_id()
    run_started_wall_clock = datetime.now(UTC)
    run_started_at = time.perf_counter()

    target_rows = int(config["dataset"]["target_rows"])
    model_specs: list[dict[str, Any]] = [dict(spec) for spec in config.get("models", [])]
    initial_model_specs: list[dict[str, Any]] = [dict(spec) for spec in model_specs]
    concurrency = int(config.get("runtime", {}).get("concurrency", 20))
    timeout_seconds = float(config.get("runtime", {}).get("request_timeout_seconds", 45))
    mock_fallback_on_failure = bool(config.get("runtime", {}).get("mock_fallback_on_failure", False))
    mock_mode = _resolve_mock_mode(config, force_live)
    provider_settings = _provider_runtime_settings(config, model_specs)
    provider_health_policy = ProviderHealthPolicy(provider_settings)

    dataset_path, generated_rows = generate_dataset(total_rows=target_rows)
    dataset_rows = [asdict(row) for row in generated_rows]

    prompt_versions = ["base", "improved"]
    all_results: list[EvalResult] = []
    all_errors: list[ModelError] = []

    for version in prompt_versions:
        prompt_template = _read_prompt(version)
        fallback_used = False
        prompt_started_at = time.perf_counter()

        model_specs = provider_health_policy.apply(model_specs, now=time.monotonic())

        enabled_models = [spec for spec in model_specs if spec.get("enabled", True)]
        total_tasks = len(enabled_models) * len(dataset_rows)
        progress_stride = max(1, total_tasks // 10) if total_tasks > 0 else 1

        def _format_eta(completed: int, total: int) -> str:
            if completed <= 0 or total <= 0 or completed >= total:
                return "0s"
            elapsed = max(0.0, time.perf_counter() - prompt_started_at)
            avg_per_task = elapsed / completed
            remaining = max(0.0, avg_per_task * (total - completed))
            if remaining >= 60:
                minutes = int(remaining // 60)
                seconds = int(round(remaining % 60))
                return f"{minutes}m{seconds:02d}s"
            return f"{int(round(remaining))}s"

        def on_progress(event: dict[str, Any]) -> None:
            completed = int(event.get("completed", 0))
            total = int(event.get("total", 0))
            if total <= 0:
                return
            if completed == 1 or completed == total or completed % progress_stride == 0:
                print(
                    (
                        f"[progress] prompt={version} completed={completed}/{total} "
                        f"success={event.get('success', 0)} errors={event.get('errors', 0)} "
                        f"eta={_format_eta(completed, total)} "
                        f"last={event.get('last_provider', 'unknown')}:{event.get('last_status', 'unknown')}"
                    ),
                    flush=True,
                )

        prompt_results, prompt_errors = asyncio.run(
            _evaluate_for_prompt(
                dataset_rows=dataset_rows,
                model_specs=model_specs,
                prompt_version=version,
                prompt_template=prompt_template,
                mock_mode=mock_mode,
                concurrency=concurrency,
                timeout_seconds=timeout_seconds,
                provider_runtime_settings=provider_settings,
                progress_callback=on_progress,
            )
        )

        if not prompt_results and not mock_mode and mock_fallback_on_failure:
            fallback_used = True
            fallback_results, fallback_errors = asyncio.run(
                _evaluate_for_prompt(
                    dataset_rows=dataset_rows,
                    model_specs=model_specs,
                    prompt_version=version,
                    prompt_template=prompt_template,
                    mock_mode=True,
                    concurrency=concurrency,
                    timeout_seconds=timeout_seconds,
                    provider_runtime_settings=provider_settings,
                    progress_callback=on_progress,
                )
            )
            prompt_results = fallback_results
            prompt_errors.extend(fallback_errors)

        all_results.extend(prompt_results)
        all_errors.extend(prompt_errors)
        _print_prompt_summary(
            prompt_version=version,
            total_tasks=total_tasks,
            prompt_results=prompt_results,
            prompt_errors=prompt_errors,
            fallback_used=fallback_used,
        )
        provider_health_policy.record_errors(prompt_errors, now=time.monotonic())
        model_specs = provider_health_policy.apply(model_specs, now=time.monotonic())

    summary = _group_summary(all_results)
    baseline_view = _baseline_improved(summary)

    manifest = _build_run_manifest(
        run_id=run_id,
        config_file=config_file,
        config=config,
        force_live=force_live,
        mock_mode=mock_mode,
        mock_fallback_on_failure=mock_fallback_on_failure,
        target_rows=target_rows,
        dataset_path=dataset_path,
        dataset_rows=dataset_rows,
        prompt_versions=prompt_versions,
        initial_model_specs=initial_model_specs,
        final_model_specs=model_specs,
        provider_settings=provider_settings,
        provider_health_snapshot=provider_health_policy.snapshot(now=time.monotonic()),
        run_started_wall_clock=run_started_wall_clock,
        run_started_perf=run_started_at,
        results=all_results,
        errors=all_errors,
    )
    manifest_path = _write_run_manifest(manifest)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "run_id": run_id,
        "mock_mode": mock_mode,
        "mock_fallback_on_failure": mock_fallback_on_failure,
        "target_rows": target_rows,
        "models": baseline_view,
        "failure_count": len(all_errors),
        "failures": [asdict(err) for err in all_errors],
        "manifest_path": _path_from_root(manifest_path),
        "provider_health": provider_health_policy.snapshot(now=time.monotonic()),
    }

    report_markdown = build_run_report_markdown(result_payload, manifest)
    report_outputs = write_run_report(run_id=run_id, report_markdown=report_markdown, reports_dir=REPORTS_DIR)
    result_payload["report_path"] = _path_from_root(report_outputs["report_path"])

    RESULTS_PATH.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    _write_rows_csv(all_results)

    with sqlite3.connect(DB_PATH) as conn:
        _ensure_schema(conn)
        _persist_run(conn, result_payload["run_id"], mock_mode, target_rows)
        _persist_rows(conn, result_payload["run_id"], all_results)

    _try_log_mlflow(result_payload["run_id"], summary)

    _print_run_summary(run_started_at=run_started_at, results=all_results, errors=all_errors)

    if not all_results:
        raise RuntimeError(
            "Pipeline produced zero successful model outputs; diagnostics were saved to results/manifest/report artifacts."
        )

    return result_payload
