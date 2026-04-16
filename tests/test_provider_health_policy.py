from __future__ import annotations

from models.llm_runner import ModelError, ProviderRuntimeSettings
from src.llm_reliability_engine.orchestrator import ProviderHealthPolicy


def _runtime_settings(cooldown_seconds: float = 30.0) -> dict[str, ProviderRuntimeSettings]:
    return {
        "openrouter": ProviderRuntimeSettings(
            rate_limit_per_minute=3,
            max_retries=2,
            retry_base_delay_seconds=0.5,
            retry_max_delay_seconds=8.0,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_cooldown_seconds=cooldown_seconds,
        )
    }


def test_provider_health_policy_reenables_after_cooldown() -> None:
    policy = ProviderHealthPolicy(_runtime_settings(cooldown_seconds=10.0))
    model_specs = [{"provider": "openrouter", "model": "auto", "enabled": True}]
    errors = [
        ModelError("auto", "openrouter", "base", "q1", "429 too many requests"),
        ModelError("auto", "openrouter", "base", "q2", "rate limit exceeded"),
        ModelError("auto", "openrouter", "base", "q3", "rate limit reached"),
    ]

    policy.record_errors(errors, now=100.0)

    disabled_specs = policy.apply(model_specs, now=105.0)
    assert disabled_specs[0]["enabled"] is False

    recovered_specs = policy.apply(model_specs, now=111.0)
    assert recovered_specs[0]["enabled"] is True

    snapshot = policy.snapshot()
    assert snapshot["openrouter"]["status"] == "healthy"


def test_provider_health_policy_keeps_hard_quota_disabled() -> None:
    policy = ProviderHealthPolicy(_runtime_settings(cooldown_seconds=10.0))
    model_specs = [{"provider": "openrouter", "model": "auto", "enabled": True}]
    errors = [
        ModelError("auto", "openrouter", "base", "q", "quota exceeded for metric"),
    ]

    policy.record_errors(errors, now=50.0)

    later_specs = policy.apply(model_specs, now=500.0)
    assert later_specs[0]["enabled"] is False

    snapshot = policy.snapshot()
    assert snapshot["openrouter"]["status"] == "disabled"
    assert snapshot["openrouter"]["reason"] == "hard_quota"


def test_provider_health_policy_preserves_config_disabled_models() -> None:
    policy = ProviderHealthPolicy(_runtime_settings(cooldown_seconds=10.0))
    model_specs = [{"provider": "openrouter", "model": "auto", "enabled": False}]
    errors = [
        ModelError("auto", "openrouter", "base", "q1", "429 too many requests"),
        ModelError("auto", "openrouter", "base", "q2", "429 too many requests"),
        ModelError("auto", "openrouter", "base", "q3", "429 too many requests"),
    ]

    policy.record_errors(errors, now=100.0)

    specs = policy.apply(model_specs, now=200.0)
    assert specs[0]["enabled"] is False
