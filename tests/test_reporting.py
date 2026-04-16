from __future__ import annotations

from pathlib import Path

from src.llm_reliability_engine.reporting import build_run_report_markdown, write_run_report


def test_build_run_report_markdown_contains_sections() -> None:
    payload = {
        "run_id": "run_20260416T020202Z",
        "target_rows": 4,
        "mock_mode": False,
        "mock_fallback_on_failure": True,
        "failure_count": 1,
        "manifest_path": "experiments/manifests/run_20260416T020202Z.manifest.json",
        "models": {
            "openrouter/auto": {
                "baseline": {
                    "count": 2,
                    "faithfulness": 0.8,
                    "hallucination_rate": 0.2,
                    "safety_score": 0.9,
                },
                "improved": {
                    "count": 2,
                    "faithfulness": 0.9,
                    "hallucination_rate": 0.1,
                    "safety_score": 0.95,
                },
                "delta": {
                    "faithfulness": 0.1,
                    "hallucination_rate": -0.1,
                    "safety_score": 0.05,
                },
            }
        },
        "failures": [
            {
                "error": "429 rate limit exceeded",
            }
        ],
    }
    manifest = {
        "started_at": "2026-04-16T00:00:00+00:00",
        "finished_at": "2026-04-16T00:00:05+00:00",
        "elapsed_seconds": 5.0,
    }

    report = build_run_report_markdown(payload, manifest)

    assert "# Run Report: run_20260416T020202Z" in report
    assert "## Overview" in report
    assert "## Model Comparison" in report
    assert "## Failures" in report
    assert "openrouter/auto" in report
    assert "rate_limit" in report


def test_write_run_report_writes_versioned_and_latest(tmp_path: Path) -> None:
    outputs = write_run_report(
        run_id="run_20260416T030303Z",
        report_markdown="# Report",
        reports_dir=tmp_path,
    )

    assert outputs["report_path"].exists()
    assert outputs["latest_path"].exists()
    assert outputs["report_path"].name == "run_20260416T030303Z.report.md"
    assert outputs["latest_path"].name == "latest.report.md"
    assert outputs["report_path"].read_text(encoding="utf-8") == "# Report"


def test_build_run_report_uses_manifest_success_count_when_present() -> None:
    payload = {
        "run_id": "run_20260416T040404Z",
        "target_rows": 2,
        "mock_mode": False,
        "mock_fallback_on_failure": False,
        "failure_count": 1,
        "models": {
            "openrouter/auto": {
                "baseline": {"count": 0},
                "improved": {"count": 2},
                "delta": {},
            }
        },
        "failures": [],
    }
    manifest = {
        "outcomes": {
            "success_count": 5,
        }
    }

    report = build_run_report_markdown(payload, manifest)
    assert "- Successful evaluations: 5" in report


def test_build_run_report_prefers_manifest_success_count() -> None:
    payload = {
        "run_id": "run_20260416T040404Z",
        "target_rows": 2,
        "mock_mode": False,
        "mock_fallback_on_failure": False,
        "failure_count": 0,
        "manifest_path": "experiments/manifests/run_20260416T040404Z.manifest.json",
        "models": {
            "openrouter/auto": {
                "baseline": {"count": 0},
                "improved": {"count": 2},
                "delta": {},
            }
        },
        "failures": [],
    }
    manifest = {
        "outcomes": {
            "success_count": 9,
            "error_count": 1,
        }
    }

    report = build_run_report_markdown(payload, manifest)

    assert "- Successful evaluations: 9" in report
