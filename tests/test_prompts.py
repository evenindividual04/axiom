from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_base_prompt_has_required_guardrails() -> None:
    prompt = (ROOT / "prompts" / "base_prompt.txt").read_text(encoding="utf-8")
    assert "{query}" in prompt
    assert "{context}" in prompt
    assert "Use only the provided context" in prompt
    assert "Do not invent rates" in prompt


def test_improved_prompt_is_explicit_about_safety_and_calculations() -> None:
    prompt = (ROOT / "prompts" / "improved_prompt.txt").read_text(encoding="utf-8")
    assert "If unsure, say \"I am not certain\"" in prompt
    assert "unsafe, unethical, or discriminatory" in prompt
    assert "formula used" in prompt


def test_advanced_prompt_requires_reasoning_and_refusal() -> None:
    prompt = (ROOT / "prompts" / "advanced_prompt.txt").read_text(encoding="utf-8")
    assert "Output format" in prompt
    assert "- Reasoning:" in prompt
    assert "If the context is insufficient" in prompt
