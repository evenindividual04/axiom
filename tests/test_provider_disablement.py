"""
Tests for provider disablement and error detection logic.

These tests verify that rate-limit errors and quota exhaustion are correctly
identified and that providers are disabled appropriately.
"""

from __future__ import annotations

from models.llm_runner import ModelError
from src.llm_reliability_engine.orchestrator import (
    _is_rate_limit_error,
    _is_hard_quota_error,
    _disable_rate_limited_models,
    _error_reason_bucket,
)


class TestRateLimitErrorDetection:
    """Tests for _is_rate_limit_error."""

    def test_detects_429_http_code(self) -> None:
        assert _is_rate_limit_error("HTTP 429: Too many requests")

    def test_detects_rate_limit_keyword(self) -> None:
        assert _is_rate_limit_error("rate limit exceeded")

    def test_detects_too_many_requests(self) -> None:
        assert _is_rate_limit_error("Error: too many requests received")

    def test_detects_quota_exceeded_in_rate_limit(self) -> None:
        assert _is_rate_limit_error("quota exceeded for requests")

    def test_detects_resource_exhausted(self) -> None:
        assert _is_rate_limit_error("resource_exhausted: API limit reached")

    def test_detects_insufficient_quota(self) -> None:
        assert _is_rate_limit_error("insufficient_quota")

    def test_case_insensitive(self) -> None:
        assert _is_rate_limit_error("RATE LIMIT EXCEEDED")
        assert _is_rate_limit_error("Rate Limit")

    def test_rejects_non_rate_limit_errors(self) -> None:
        assert not _is_rate_limit_error("Connection refused")
        assert not _is_rate_limit_error("Invalid API key")
        assert not _is_rate_limit_error("Server error 500")
        assert not _is_rate_limit_error("Authorization failed")


class TestHardQuotaErrorDetection:
    """Tests for _is_hard_quota_error."""

    def test_detects_quota_exceeded(self) -> None:
        assert _is_hard_quota_error("quota exceeded for metric")

    def test_detects_requests_per_day_limit(self) -> None:
        assert _is_hard_quota_error("requests per day limit reached")
        assert _is_hard_quota_error("Requests per day limit: 0")

    def test_detects_limit_zero(self) -> None:
        assert _is_hard_quota_error("limit: 0")

    def test_detects_free_tier_requests_exhausted(self) -> None:
        assert _is_hard_quota_error("free_tier_requests, limit: 0")

    def test_detects_free_tier_input_token_limit(self) -> None:
        assert _is_hard_quota_error("free_tier_input_token_count, limit: 0")

    def test_detects_insufficient_quota(self) -> None:
        assert _is_hard_quota_error("insufficient_quota for operation")

    def test_detects_billing_details_required(self) -> None:
        assert _is_hard_quota_error("Billing details must be updated")

    def test_case_insensitive(self) -> None:
        assert _is_hard_quota_error("QUOTA EXCEEDED")
        assert _is_hard_quota_error("Quota Exceeded")

    def test_rejects_transient_rate_limits(self) -> None:
        # Transient rate limits should not be detected as hard quota
        assert not _is_hard_quota_error("Rate limited temporarily")
        assert not _is_hard_quota_error("429 Too Many Requests")


class TestDisableRateLimitedModels:
    """Tests for _disable_rate_limited_models."""

    def test_disables_provider_on_hard_quota_error(self) -> None:
        model_specs = [
            {"model": "gpt-4", "provider": "openai", "enabled": True},
            {"model": "text-davinci", "provider": "openai", "enabled": True},
        ]
        errors = [
            ModelError(
                model="gpt-4",
                provider="openai",
                prompt_version="base",
                question="test",
                error="quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_*",
            )
        ]

        updated = _disable_rate_limited_models(model_specs, errors)

        # All OpenAI models should be disabled
        assert all(not spec.get("enabled", True) for spec in updated if spec["provider"] == "openai")

    def test_disables_provider_after_3_rate_limit_errors(self) -> None:
        model_specs = [
            {"model": "llama", "provider": "groq", "enabled": True},
        ]
        errors = [
            ModelError("llama", "groq", "base", "q1", "429 too many requests"),
            ModelError("llama", "groq", "base", "q2", "rate limit exceeded"),
            ModelError("llama", "groq", "base", "q3", "Rate limit reached"),
        ]

        updated = _disable_rate_limited_models(model_specs, errors)

        # After 3 rate limit errors, Groq should be disabled
        assert not updated[0]["enabled"]

    def test_does_not_disable_on_fewer_than_3_rate_limits(self) -> None:
        model_specs = [
            {"model": "test-model", "provider": "provider-a", "enabled": True},
        ]
        errors = [
            ModelError("test-model", "provider-a", "base", "q1", "429 error"),
            ModelError("test-model", "provider-a", "base", "q2", "rate limit"),
        ]

        updated = _disable_rate_limited_models(model_specs, errors)

        # Only 2 errors; should still be enabled
        assert updated[0]["enabled"] is True

    def test_multiple_providers_selective_disablement(self) -> None:
        model_specs = [
            {"model": "model-a", "provider": "provider-a", "enabled": True},
            {"model": "model-b", "provider": "provider-b", "enabled": True},
            {"model": "model-c", "provider": "provider-c", "enabled": True},
        ]
        errors = [
            # Provider A has hard quota error → disable immediately
            ModelError("model-a", "provider-a", "base", "q", "quota exceeded"),
            # Provider B has 3 rate limit errors → disable
            ModelError("model-b", "provider-b", "base", "q1", "429"),
            ModelError("model-b", "provider-b", "base", "q2", "rate limit"),
            ModelError("model-b", "provider-b", "base", "q3", "rate limit"),
            # Provider C has only 1 rate limit error → keep enabled
            ModelError("model-c", "provider-c", "base", "q1", "rate limit"),
        ]

        updated = _disable_rate_limited_models(model_specs, errors)

        assert not updated[0]["enabled"]  # provider-a disabled
        assert not updated[1]["enabled"]  # provider-b disabled
        assert updated[2]["enabled"] is True  # provider-c still enabled

    def test_preserves_spec_details(self) -> None:
        model_specs = [
            {"model": "gpt-4", "provider": "openai", "enabled": True, "custom_field": "value"},
        ]
        errors = [
            ModelError("gpt-4", "openai", "base", "q", "quota exceeded"),
        ]

        updated = _disable_rate_limited_models(model_specs, errors)

        # Custom fields should be preserved
        assert updated[0]["custom_field"] == "value"
        assert updated[0]["model"] == "gpt-4"

    def test_handles_empty_errors(self) -> None:
        model_specs = [
            {"model": "test", "provider": "provider-a", "enabled": True},
        ]
        errors = []

        updated = _disable_rate_limited_models(model_specs, errors)

        # No changes if no errors
        assert updated == model_specs

    def test_handles_empty_model_specs(self) -> None:
        model_specs: list[dict] = []
        errors = [
            ModelError("model", "provider", "base", "q", "error"),
        ]

        updated = _disable_rate_limited_models(model_specs, errors)

        assert updated == []

    def test_case_insensitive_provider_matching(self) -> None:
        model_specs = [
            {"model": "test", "provider": "OpenAI", "enabled": True},
        ]
        errors = [
            ModelError("test", "openai", "base", "q", "quota exceeded"),
        ]

        updated = _disable_rate_limited_models(model_specs, errors)

        # Should match despite case difference
        assert not updated[0]["enabled"]


class TestErrorReasonBucket:
    """Tests for _error_reason_bucket."""

    def test_categorizes_circuit_breaker_open(self) -> None:
        assert _error_reason_bucket("circuit breaker open") == "circuit_breaker_open"

    def test_categorizes_hard_quota(self) -> None:
        assert _error_reason_bucket("quota exceeded") == "hard_quota"
        assert _error_reason_bucket("limit: 0") == "hard_quota"

    def test_categorizes_rate_limit(self) -> None:
        assert _error_reason_bucket("429 too many requests") == "rate_limit"
        assert _error_reason_bucket("rate limit exceeded") == "rate_limit"

    def test_categorizes_timeout(self) -> None:
        assert _error_reason_bucket("Request timeout") == "timeout"
        assert _error_reason_bucket("Operation timed out") == "timeout"

    def test_categorizes_connection(self) -> None:
        assert _error_reason_bucket("Connection refused") == "connection"

    def test_defaults_to_other(self) -> None:
        assert _error_reason_bucket("Unknown error") == "other"
        assert _error_reason_bucket("") == "other"

    def test_case_insensitive(self) -> None:
        assert _error_reason_bucket("CIRCUIT BREAKER OPEN") == "circuit_breaker_open"
        assert _error_reason_bucket("TIMEOUT") == "timeout"
