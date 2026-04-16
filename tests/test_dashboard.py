from __future__ import annotations

import pandas as pd

from dashboard import app


def test_build_summary_cards_uses_manifest_and_rows() -> None:
    results = {
        "run_id": "run_123",
        "mock_mode": True,
        "mock_fallback_on_failure": False,
        "failure_count": 2,
        "manifest_path": "experiments/manifests/run_123.manifest.json",
        "report_path": "experiments/reports/run_123.report.md",
        "models": {
            "openrouter/auto": {
                "baseline": {"count": 2},
                "improved": {"count": 2},
                "delta": {"hallucination_rate": -0.1, "faithfulness": 0.2, "safety_score": 0.0},
            }
        },
        "outcomes": {"success_count": 4, "error_count": 0, "success_rate": 100.0},
        "provider_health": {"openrouter": {"status": "healthy", "reason": ""}},
    }
    rows_df = pd.DataFrame(
        [
            {"model": "openrouter/auto", "prompt_version": "base", "failure_type": "correct", "hallucination_rate": 0.2, "faithfulness": 0.8, "answer_relevancy": 0.1, "safety_score": 1.0, "latency_ms": 12.0},
            {"model": "openrouter/auto", "prompt_version": "improved", "failure_type": "correct", "hallucination_rate": 0.1, "faithfulness": 0.9, "answer_relevancy": 0.2, "safety_score": 1.0, "latency_ms": 10.0},
        ]
    )

    summary = app.build_summary_cards(results, rows_df)

    assert summary["run_id"] == "run_123"
    assert summary["successful_evaluations"] == 4
    assert summary["failure_count"] == 2
    assert summary["top_model"] == "openrouter/auto"


def test_build_provider_health_frame_handles_missing_snapshot() -> None:
    frame = app.build_provider_health_frame({})
    assert list(frame.columns) == ["provider", "status", "reason", "disabled_until", "last_error", "transient_error_count"]
    assert frame.empty


def test_build_delta_frame_sorts_by_hallucination_improvement() -> None:
    results = {
        "models": {
            "model-a": {
                "delta": {"hallucination_rate": -0.2, "faithfulness": 0.1, "safety_score": 0.0},
                "baseline": {"count": 2, "faithfulness": 0.6, "hallucination_rate": 0.3, "safety_score": 0.9},
                "improved": {"count": 2, "faithfulness": 0.7, "hallucination_rate": 0.1, "safety_score": 0.9},
            },
            "model-b": {
                "delta": {"hallucination_rate": -0.05, "faithfulness": -0.1, "safety_score": 0.0},
                "baseline": {"count": 2, "faithfulness": 0.8, "hallucination_rate": 0.2, "safety_score": 0.95},
                "improved": {"count": 2, "faithfulness": 0.7, "hallucination_rate": 0.15, "safety_score": 0.95},
            },
        }
    }

    frame = app.build_delta_frame(results)

    assert list(frame["model"]) == ["model-a", "model-b"]
    assert frame.iloc[0]["hallucination_delta"] == -0.2


def test_format_health_status_marks_disabled() -> None:
    assert app.format_health_status({"status": "disabled", "reason": "hard_quota"}) == "disabled"
    assert app.format_health_status({"status": "cooling_down", "reason": "rate_limit"}) == "cooling_down"
    assert app.format_health_status({}) == "healthy"
