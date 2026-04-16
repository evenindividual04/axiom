from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from evals.pipeline import EvalResult
from models.llm_runner import ModelError, ProviderRuntimeSettings
from src.llm_reliability_engine import orchestrator


def _sample_result() -> EvalResult:
    return EvalResult(
        question="What is EMI?",
        model="openrouter/auto",
        prompt_version="base",
        output="EMI is monthly payment.",
        answer_relevancy=0.9,
        faithfulness=0.8,
        hallucination_rate=0.1,
        safety_score=1.0,
        failure_type="none",
        latency_ms=1200.0,
        token_count=42,
    )


def test_enabled_provider_and_model_helpers() -> None:
    specs = [
        {"provider": "OpenRouter", "model": "auto", "enabled": True},
        {"provider": "openrouter", "model": "auto", "enabled": True},
        {"provider": "groq", "model": "llama", "enabled": False},
    ]

    providers = orchestrator._enabled_providers(specs)
    models = orchestrator._enabled_models(specs)

    assert providers == ["openrouter"]
    assert models == ["openrouter/auto"]


def test_build_run_manifest_contains_expected_sections() -> None:
    run_id = "run_20260416T000000Z"
    config_file = Path("/tmp/config.yaml")
    config = {"runtime": {"mock_mode": False}, "dataset": {"target_rows": 2}}
    provider_settings = {
        "openrouter": ProviderRuntimeSettings(
            rate_limit_per_minute=3,
            max_retries=2,
            retry_base_delay_seconds=0.5,
            retry_max_delay_seconds=8.0,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_cooldown_seconds=240.0,
        )
    }

    initial_specs = [
        {"provider": "openrouter", "model": "auto", "enabled": True},
        {"provider": "groq", "model": "llama", "enabled": True},
    ]
    final_specs = [
        {"provider": "openrouter", "model": "auto", "enabled": True},
        {"provider": "groq", "model": "llama", "enabled": False},
    ]
    dataset_rows = [
        {"test_type": "happy_path", "domain": "lending"},
        {"test_type": "adversarial", "domain": "financial_ops"},
    ]

    manifest = orchestrator._build_run_manifest(
        run_id=run_id,
        config_file=config_file,
        config=config,
        force_live=True,
        mock_mode=False,
        mock_fallback_on_failure=True,
        target_rows=2,
        dataset_path=Path("/tmp/dataset.csv"),
        dataset_rows=dataset_rows,
        prompt_versions=["base", "improved"],
        initial_model_specs=initial_specs,
        final_model_specs=final_specs,
        provider_settings=provider_settings,
        run_started_wall_clock=datetime.now(UTC),
        run_started_perf=0.0,
        results=[_sample_result()],
        errors=[
            ModelError(
                model="llama",
                provider="groq",
                prompt_version="base",
                question="q",
                error="rate limit exceeded",
            )
        ],
    )

    assert manifest["run_id"] == run_id
    assert manifest["runtime"]["force_live"] is True
    assert manifest["dataset"]["generated_rows"] == 2
    assert manifest["providers"]["disabled_during_run"] == ["groq"]
    assert manifest["outcomes"]["success_count"] == 1
    assert manifest["outcomes"]["error_count"] == 1
    assert "manifests" not in manifest["artifacts"]


def test_write_run_manifest_creates_versioned_and_latest_files(tmp_path: Path) -> None:
    manifest = {
        "run_id": "run_20260416T010101Z",
        "hello": "world",
    }
    expected = tmp_path / "run_20260416T010101Z.manifest.json"
    latest = tmp_path / "latest.manifest.json"

    original = orchestrator.MANIFESTS_DIR
    try:
        orchestrator.MANIFESTS_DIR = tmp_path
        written_path = orchestrator._write_run_manifest(manifest)
    finally:
        orchestrator.MANIFESTS_DIR = original

    assert written_path == expected
    assert expected.exists()
    assert latest.exists()
    assert "world" in expected.read_text(encoding="utf-8")


def test_path_from_root_handles_external_paths() -> None:
    external = Path("/tmp/somewhere/outside.json")
    assert orchestrator._path_from_root(external) == str(external)


def test_new_run_id_is_unique_even_for_same_timestamp() -> None:
    fixed = datetime(2026, 4, 16, 12, 0, 0, 123456, tzinfo=UTC)
    run_id_a = orchestrator._new_run_id(fixed)
    run_id_b = orchestrator._new_run_id(fixed)

    assert run_id_a != run_id_b
    assert run_id_a.startswith("run_20260416T120000123456Z_")
    assert run_id_b.startswith("run_20260416T120000123456Z_")
