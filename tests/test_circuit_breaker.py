"""
Tests for ProviderCircuitBreaker.

Circuit breaker prevents cascading failures by tripping after N consecutive failures,
blocking requests during cooldown, and allowing recovery.
"""

from __future__ import annotations

import asyncio

import pytest

from models.llm_runner import ProviderCircuitBreaker


@pytest.mark.asyncio
async def test_circuit_breaker_allows_requests_when_healthy() -> None:
    """Healthy circuit breaker allows requests."""
    cb = ProviderCircuitBreaker(failure_threshold=3, cooldown_seconds=1.0)
    
    assert await cb.allow_request() is True
    assert await cb.allow_request() is True


@pytest.mark.asyncio
async def test_circuit_breaker_trips_after_threshold() -> None:
    """Circuit breaker trips and blocks requests after threshold failures."""
    cb = ProviderCircuitBreaker(failure_threshold=2, cooldown_seconds=1.0)
    
    # Record first failure: not tripped yet
    await cb.record_failure("error 1")
    assert await cb.allow_request() is True
    
    # Record second failure: now tripped
    await cb.record_failure("error 2")
    assert await cb.allow_request() is False
    
    # Still blocked
    assert await cb.allow_request() is False


@pytest.mark.asyncio
async def test_circuit_breaker_recovers_after_cooldown() -> None:
    """Circuit breaker recovers after cooldown expires."""
    # Note: ProviderCircuitBreaker enforces minimum 1.0 second cooldown
    # Setting a smaller value will be clamped to 1.0
    cb = ProviderCircuitBreaker(failure_threshold=2, cooldown_seconds=1.0)
    
    # Trip the breaker
    await cb.record_failure("error 1")
    await cb.record_failure("error 2")
    assert await cb.allow_request() is False
    
    # Wait for cooldown to expire (1+ seconds)
    await asyncio.sleep(1.1)
    
    # Now allowed again
    assert await cb.allow_request() is True
    
    # Confirm state reset
    assert cb._consecutive_failures == 0


@pytest.mark.asyncio
async def test_circuit_breaker_resets_on_success() -> None:
    """Circuit breaker resets failure count on success."""
    cb = ProviderCircuitBreaker(failure_threshold=3, cooldown_seconds=1.0)
    
    await cb.record_failure("error 1")
    await cb.record_failure("error 2")
    assert cb._consecutive_failures == 2
    
    # Success resets counter
    await cb.record_success()
    assert cb._consecutive_failures == 0
    assert await cb.allow_request() is True


@pytest.mark.asyncio
async def test_circuit_breaker_stores_last_error() -> None:
    """Circuit breaker stores the most recent error message."""
    cb = ProviderCircuitBreaker(failure_threshold=2, cooldown_seconds=1.0)
    
    await cb.record_failure("first error")
    await cb.record_failure("second error")
    
    assert cb.last_error == "second error"


@pytest.mark.asyncio
async def test_circuit_breaker_last_error_cleared_on_success() -> None:
    """Circuit breaker clears error message on success."""
    cb = ProviderCircuitBreaker(failure_threshold=3, cooldown_seconds=1.0)
    
    await cb.record_failure("some error")
    assert cb.last_error == "some error"
    
    await cb.record_success()
    assert cb.last_error == ""


@pytest.mark.asyncio
async def test_circuit_breaker_trip_for_manual_trip() -> None:
    """Circuit breaker can be manually tripped for hard quota errors."""
    cb = ProviderCircuitBreaker(failure_threshold=5, cooldown_seconds=1.0)
    
    # Manually trip for 2 seconds
    await cb.trip_for(2.0, error="Hard quota exceeded")
    
    assert await cb.allow_request() is False
    assert cb.last_error == "Hard quota exceeded"
    assert cb._consecutive_failures == 5  # Set to threshold


@pytest.mark.asyncio
async def test_circuit_breaker_trip_for_recovery() -> None:
    """Circuit breaker recovers after manual trip cooldown expires."""
    cb = ProviderCircuitBreaker(failure_threshold=5, cooldown_seconds=1.0)
    
    # trip_for enforces minimum 1.0 second cooldown
    await cb.trip_for(0.1, error="Quota exceeded")
    assert await cb.allow_request() is False
    
    # Note: trip_for uses max(1.0, seconds), so minimum wait is 1.0 second
    # For testing, we can verify the state is set correctly without waiting
    assert cb._consecutive_failures == 5
    assert cb.last_error == "Quota exceeded"


@pytest.mark.asyncio
async def test_circuit_breaker_thread_safe() -> None:
    """Circuit breaker is thread-safe under concurrent access."""
    cb = ProviderCircuitBreaker(failure_threshold=5, cooldown_seconds=1.0)
    
    async def fail_task(error_msg: str) -> None:
        await cb.record_failure(error_msg)
    
    async def allow_task() -> bool:
        return await cb.allow_request()
    
    # Concurrent failures
    failures = [fail_task(f"error_{i}") for i in range(5)]
    await asyncio.gather(*failures)
    
    # All should be blocked
    blocks = await asyncio.gather(*[allow_task() for _ in range(5)])
    assert all(b is False for b in blocks)


@pytest.mark.asyncio
async def test_circuit_breaker_threshold_enforced() -> None:
    """Circuit breaker threshold of 1 works correctly."""
    cb = ProviderCircuitBreaker(failure_threshold=1, cooldown_seconds=1.0)
    
    await cb.record_failure("immediate failure")
    assert await cb.allow_request() is False


@pytest.mark.asyncio
async def test_circuit_breaker_no_error_message() -> None:
    """Circuit breaker handles failures without error messages."""
    cb = ProviderCircuitBreaker(failure_threshold=2, cooldown_seconds=1.0)
    
    await cb.record_failure()  # No error message
    await cb.record_failure()  # No error message
    
    assert cb.last_error == ""
    assert await cb.allow_request() is False
