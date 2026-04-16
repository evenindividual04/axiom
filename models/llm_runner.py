from __future__ import annotations

import asyncio
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable
from collections import deque


@dataclass(frozen=True)
class ModelOutput:
    model: str
    provider: str
    prompt_version: str
    question: str
    context: str
    ground_truth: str
    output: str
    latency_ms: float
    token_count: int


@dataclass(frozen=True)
class ModelError:
    model: str
    provider: str
    prompt_version: str
    question: str
    error: str


@dataclass(frozen=True)
class ProviderRuntimeSettings:
    rate_limit_per_minute: int
    max_retries: int
    retry_base_delay_seconds: float
    retry_max_delay_seconds: float
    circuit_breaker_failure_threshold: int
    circuit_breaker_cooldown_seconds: float


class ProviderRateLimiter:
    def __init__(self, rate_limit_per_minute: int):
        self.rate_limit_per_minute = max(1, rate_limit_per_minute)
        self._events: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            while self._events and now - self._events[0] >= 60:
                self._events.popleft()

            if len(self._events) >= self.rate_limit_per_minute:
                sleep_for = max(0.0, 60 - (now - self._events[0]))
                await asyncio.sleep(sleep_for)
                now = time.monotonic()
                while self._events and now - self._events[0] >= 60:
                    self._events.popleft()

            self._events.append(time.monotonic())


class ProviderCircuitBreaker:
    def __init__(self, failure_threshold: int, cooldown_seconds: float):
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(1.0, cooldown_seconds)
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._last_error = ""
        self._lock = asyncio.Lock()

    @property
    def last_error(self) -> str:
        return self._last_error

    async def allow_request(self) -> bool:
        async with self._lock:
            if self._open_until <= 0:
                return True
            if time.monotonic() >= self._open_until:
                self._open_until = 0.0
                self._consecutive_failures = 0
                return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            self._consecutive_failures = 0
            self._open_until = 0.0
            self._last_error = ""

    async def record_failure(self, error: str = "") -> None:
        async with self._lock:
            self._consecutive_failures += 1
            if error:
                self._last_error = error
            if self._consecutive_failures >= self.failure_threshold:
                self._open_until = time.monotonic() + self.cooldown_seconds

    async def trip_for(self, seconds: float, error: str = "") -> None:
        async with self._lock:
            self._consecutive_failures = self.failure_threshold
            self._open_until = time.monotonic() + max(1.0, seconds)
            if error:
                self._last_error = error


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    message = str(exc).lower()
    if re.search(r"\b429\b", message) or re.search(r"\b5\d\d\b", message):
        return True

    retry_tokens = [
        "rate limit",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection",
    ]
    return any(token in message for token in retry_tokens)


def _is_non_retryable_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    hard_quota_tokens = [
        "quota exceeded",
        "requests per day",
        "free_tier_requests, limit: 0",
        "free_tier_input_token_count, limit: 0",
        "insufficient_quota",
        "billing details",
    ]
    return any(token in message for token in hard_quota_tokens)


def _mock_response(*, prompt_version: str, question: str, ground_truth: str, test_type: str) -> str:
    """Generate realistic financial domain mock responses for fallback."""
    base = ground_truth
    lower_q = question.lower()
    
    # Domain-specific variations for adversarial/edge cases
    adversarial_templates = [
        "It's possible depending on your relationship with the lender and negotiation.",
        "The bank may consider exceptions on a case-by-case basis.",
        "This depends on the specific terms of your loan agreement.",
        "Generally the terms are non-negotiable, but your bank may offer alternatives.",
    ]
    
    edge_case_templates = [
        "The outcome may vary based on the specific lender's policies.",
        "This scenario is not explicitly covered in standard policies; contact your lender.",
        "The result depends on additional factors not mentioned in your question.",
        "Your specific situation may have unique considerations.",
    ]
    
    hallucination_templates = [
        f"{base} Additionally, the Reserve Bank provides oversight on such transactions.",
        f"{base} Most lenders align with regulatory guidelines on this matter.",
        f"{base} This is a standard industry practice across most institutions.",
    ]
    
    uncertainty_templates = [
        f"I cannot provide a definitive answer without knowing your specific lender's policies. Based on general practices: {base}",
        f"The answer may vary by lender. Typically: {base}. Verify with your bank.",
        f"While general guidelines suggest {base}, individual circumstances may differ.",
    ]
    
    # Base prompt: introduce controlled variations
    if prompt_version == "base":
        halluc_roll = random.random()
        if halluc_roll < 0.15 and test_type == "adversarial":
            return random.choice(adversarial_templates)
        elif halluc_roll < 0.20 and test_type == "edge_case":
            return random.choice(edge_case_templates)
        elif halluc_roll < 0.25:
            return random.choice(hallucination_templates)
        return base
    
    # Improved prompt: more conservative, adds uncertainty hedging
    if prompt_version == "improved":
        if random.random() < 0.10:
            return random.choice(uncertainty_templates)
        return base
    
    # Advanced prompt: structured reasoning for numeric questions
    if prompt_version == "advanced":
        if "emi" in lower_q or "interest" in lower_q or "loan" in lower_q:
            calculations = [
                f"Based on the loan parameters provided: {base}",
                f"Using standard amortization: {base}",
                f"Calculated as follows: {base}",
            ]
            return random.choice(calculations)
        return base
    
    return base


async def _run_one(
    *,
    model: str,
    provider: str,
    prompt_version: str,
    question: str,
    context: str,
    ground_truth: str,
    test_type: str,
    prompt_template: str,
    mock_mode: bool,
    timeout_seconds: float,
    rate_limiter: ProviderRateLimiter | None,
) -> ModelOutput:
    start = time.perf_counter()

    if mock_mode:
        await asyncio.sleep(random.uniform(0.01, 0.04))
        output = _mock_response(
            prompt_version=prompt_version,
            question=question,
            ground_truth=ground_truth,
            test_type=test_type,
        )
    else:
        if rate_limiter is not None:
            await rate_limiter.acquire()
        output = await asyncio.wait_for(
            _run_live(
                provider=provider,
                model=model,
                question=question,
                context=context,
                prompt_template=prompt_template,
            ),
            timeout=timeout_seconds,
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    token_count = len((question + " " + output).split())

    return ModelOutput(
        model=model,
        provider=provider,
        prompt_version=prompt_version,
        question=question,
        context=context,
        ground_truth=ground_truth,
        output=output,
        latency_ms=latency_ms,
        token_count=token_count,
    )


async def _run_live(*, provider: str, model: str, question: str, context: str, prompt_template: str) -> str:
    prompt = prompt_template.format(query=question, context=context)

    if provider in {"openai", "groq", "openrouter", "gemini", "zai"}:
        from openai import AsyncOpenAI

        api_key = ""
        base_url: str | None = None

        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        elif provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
            base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        elif provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        elif provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
            base_url = os.environ.get("GEMINI_OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        elif provider == "zai":
            api_key = os.environ.get("ZAI_API_KEY", "")
            base_url = os.environ.get("ZAI_BASE_URL", "").strip() or None

        if not api_key:
            raise RuntimeError(f"API key is required for provider={provider} when mock_mode=false")
        if provider == "zai" and not base_url:
            raise RuntimeError("ZAI_BASE_URL is required for provider=zai when mock_mode=false")

        async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or ""

    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when mock_mode=false")

        async with AsyncAnthropic(api_key=api_key) as client:
            resp = await client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in resp.content if hasattr(block, "text"))

    if provider == "ollama":
        from ollama import AsyncClient

        client = AsyncClient()
        resp: Any = await client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"]

    raise ValueError(f"Unsupported provider: {provider}")


async def run_models(
    *,
    dataset: list[dict[str, Any]],
    model_specs: list[dict[str, Any]],
    prompt_version: str,
    prompt_template: str,
    mock_mode: bool,
    concurrency: int,
    timeout_seconds: float,
    provider_runtime_settings: dict[str, ProviderRuntimeSettings],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[ModelOutput], list[ModelError]]:
    sem = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task[ModelOutput | ModelError]] = []
    provider_limiters: dict[str, ProviderRateLimiter] = {}
    provider_circuits: dict[str, ProviderCircuitBreaker] = {}

    for provider_name, settings in provider_runtime_settings.items():
        provider_limiters[provider_name] = ProviderRateLimiter(settings.rate_limit_per_minute)
        provider_circuits[provider_name] = ProviderCircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
        )

    async def bounded(spec: dict[str, Any], row: dict[str, Any]) -> ModelOutput | ModelError:
        async with sem:
            model_name = str(spec.get("name", "unknown"))
            provider_name = str(spec.get("provider", "unknown")).strip().lower()
            runtime = provider_runtime_settings.get(provider_name)
            if runtime is None:
                runtime = ProviderRuntimeSettings(
                    rate_limit_per_minute=60,
                    max_retries=2,
                    retry_base_delay_seconds=0.5,
                    retry_max_delay_seconds=8.0,
                    circuit_breaker_failure_threshold=5,
                    circuit_breaker_cooldown_seconds=30.0,
                )
                provider_runtime_settings[provider_name] = runtime

            if provider_name not in provider_limiters:
                provider_limiters[provider_name] = ProviderRateLimiter(runtime.rate_limit_per_minute)
            if provider_name not in provider_circuits:
                provider_circuits[provider_name] = ProviderCircuitBreaker(
                    failure_threshold=runtime.circuit_breaker_failure_threshold,
                    cooldown_seconds=runtime.circuit_breaker_cooldown_seconds,
                )

            circuit = provider_circuits.get(provider_name)

            last_error: Exception | None = None
            for attempt in range(runtime.max_retries + 1):
                if circuit is not None and not await circuit.allow_request():
                    return ModelError(
                        model=model_name,
                        provider=provider_name,
                        prompt_version=prompt_version,
                        question=str(row.get("question", "unknown")),
                        error=(
                            "Circuit breaker open for provider"
                            + (f"; last_error={circuit.last_error}" if circuit.last_error else "")
                        ),
                    )
                try:
                    result = await _run_one(
                        model=model_name,
                        provider=provider_name,
                        prompt_version=prompt_version,
                        question=row["question"],
                        context=row["context"],
                        ground_truth=row["ground_truth"],
                        test_type=row.get("test_type", "happy_path"),
                        prompt_template=prompt_template,
                        mock_mode=mock_mode,
                        timeout_seconds=timeout_seconds,
                        rate_limiter=provider_limiters.get(provider_name),
                    )
                    if circuit is not None:
                        await circuit.record_success()
                    return result
                except Exception as exc:
                    last_error = exc
                    if _is_non_retryable_quota_error(exc):
                        if circuit is not None:
                            await circuit.trip_for(
                                seconds=max(runtime.circuit_breaker_cooldown_seconds, 300.0),
                                error=str(exc),
                            )
                        break

                    if attempt >= runtime.max_retries or not _is_retryable_error(exc):
                        if circuit is not None:
                            await circuit.record_failure(str(exc))
                        break
                    sleep_for = min(
                        runtime.retry_max_delay_seconds,
                        runtime.retry_base_delay_seconds * (2 ** attempt),
                    ) + random.uniform(0, 0.2)
                    await asyncio.sleep(sleep_for)

            return ModelError(
                model=model_name,
                provider=provider_name,
                prompt_version=prompt_version,
                question=str(row.get("question", "unknown")),
                error=str(last_error) if last_error else "Unknown model execution failure",
            )

    for spec in model_specs:
        if not spec.get("enabled", True):
            continue
        for row in dataset:
            tasks.append(asyncio.create_task(bounded(spec, row)))

    results: list[ModelOutput | ModelError] = []
    total = len(tasks)
    completed = 0
    success_count = 0
    error_count = 0

    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        completed += 1

        if isinstance(result, ModelError):
            error_count += 1
            last_provider = result.provider
            last_status = "error"
        else:
            success_count += 1
            last_provider = result.provider
            last_status = "success"

        if progress_callback is not None:
            progress_callback(
                {
                    "total": total,
                    "completed": completed,
                    "success": success_count,
                    "errors": error_count,
                    "last_provider": last_provider,
                    "last_status": last_status,
                }
            )

    outputs: list[ModelOutput] = []
    errors: list[ModelError] = []
    for result in results:
        if isinstance(result, ModelError):
            errors.append(result)
        else:
            outputs.append(result)

    return outputs, errors
