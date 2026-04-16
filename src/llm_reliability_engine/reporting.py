from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return str(value)


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "n/a"
    if not isinstance(value, (float, int)):
        return str(value)
    if value > 0:
        return f"+{float(value):.4f}"
    return f"{float(value):.4f}"


def build_run_report_markdown(result_payload: dict[str, Any], manifest: dict[str, Any] | None = None) -> str:
    run_id = str(result_payload.get("run_id", "unknown"))
    target_rows = int(result_payload.get("target_rows", 0))
    failure_count = int(result_payload.get("failure_count", 0))
    mock_mode = bool(result_payload.get("mock_mode", False))
    mock_fallback = bool(result_payload.get("mock_fallback_on_failure", False))
    models = result_payload.get("models", {})

    success_count: int | None = None
    if isinstance(manifest, dict):
        outcomes = manifest.get("outcomes", {}) if isinstance(manifest.get("outcomes"), dict) else {}
        maybe_success = outcomes.get("success_count")
        if isinstance(maybe_success, int):
            success_count = maybe_success
    if success_count is None:
        success_count = 0
        for model_data in models.values():
            if not isinstance(model_data, dict):
                continue
            baseline = model_data.get("baseline", {}) if isinstance(model_data.get("baseline"), dict) else {}
            improved = model_data.get("improved", {}) if isinstance(model_data.get("improved"), dict) else {}
            success_count += int(baseline.get("count", 0)) + int(improved.get("count", 0))

    started_at = "n/a"
    finished_at = "n/a"
    elapsed_seconds = "n/a"
    if isinstance(manifest, dict):
        started_at = str(manifest.get("started_at", "n/a"))
        finished_at = str(manifest.get("finished_at", "n/a"))
        elapsed = manifest.get("elapsed_seconds")
        if isinstance(elapsed, (float, int)):
            elapsed_seconds = f"{float(elapsed):.2f}s"

    lines: list[str] = []
    lines.append(f"# Run Report: {run_id}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now(UTC).isoformat()}")
    lines.append(f"- Mock mode: {str(mock_mode).lower()}")
    lines.append(f"- Mock fallback on failure: {str(mock_fallback).lower()}")
    lines.append(f"- Target rows: {target_rows}")
    lines.append(f"- Successful evaluations: {success_count}")
    lines.append(f"- Failed evaluations: {failure_count}")
    lines.append(f"- Started at: {started_at}")
    lines.append(f"- Finished at: {finished_at}")
    lines.append(f"- Elapsed: {elapsed_seconds}")
    lines.append("")

    lines.append("## Model Comparison")
    lines.append("")
    lines.append("| Model | Baseline Faithfulness | Improved Faithfulness | Delta Faithfulness | Baseline Hallucination | Improved Hallucination | Delta Hallucination | Baseline Safety | Improved Safety | Delta Safety |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for model_name in sorted(models.keys()):
        data = models.get(model_name, {})
        baseline = data.get("baseline", {}) if isinstance(data, dict) else {}
        improved = data.get("improved", {}) if isinstance(data, dict) else {}
        delta = data.get("delta", {}) if isinstance(data, dict) else {}

        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    _fmt_metric(baseline.get("faithfulness")),
                    _fmt_metric(improved.get("faithfulness")),
                    _fmt_delta(delta.get("faithfulness")),
                    _fmt_metric(baseline.get("hallucination_rate")),
                    _fmt_metric(improved.get("hallucination_rate")),
                    _fmt_delta(delta.get("hallucination_rate")),
                    _fmt_metric(baseline.get("safety_score")),
                    _fmt_metric(improved.get("safety_score")),
                    _fmt_delta(delta.get("safety_score")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Failures")
    lines.append("")

    failures = result_payload.get("failures", [])
    if isinstance(failures, list) and failures:
        reason_counts: dict[str, int] = {}
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            error = str(failure.get("error", "other")).lower()
            bucket = "other"
            if "circuit breaker open" in error:
                bucket = "circuit_breaker_open"
            elif "quota exceeded" in error or "insufficient_quota" in error or "limit: 0" in error:
                bucket = "hard_quota"
            elif "rate limit" in error or "429" in error:
                bucket = "rate_limit"
            elif "timeout" in error:
                bucket = "timeout"
            elif "connection" in error:
                bucket = "connection"
            reason_counts[bucket] = reason_counts.get(bucket, 0) + 1

        lines.append("| Failure Reason | Count |")
        lines.append("|---|---:|")
        for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("No failures recorded.")

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- results.json: experiments/results.json")
    lines.append("- row_results.csv: experiments/row_results.csv")
    lines.append(f"- manifest: {result_payload.get('manifest_path', 'n/a')}")

    return "\n".join(lines)


def write_run_report(*, run_id: str, report_markdown: str, reports_dir: Path) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{run_id}.report.md"
    latest_path = reports_dir / "latest.report.md"

    report_path.write_text(report_markdown, encoding="utf-8")
    latest_path.write_text(report_markdown, encoding="utf-8")

    return {
        "report_path": report_path,
        "latest_path": latest_path,
    }
