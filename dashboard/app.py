from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "experiments" / "results.json"
ROWS_PATH = ROOT / "experiments" / "row_results.csv"
MANIFESTS_DIR = ROOT / "experiments" / "manifests"
REPORTS_DIR = ROOT / "experiments" / "reports"

THEME_CSS = """
<style>
    :root {
        --bg: #0a0f14;
        --panel: #111822;
        --panel-2: #172230;
        --text: #edf2f7;
        --muted: #9fb0c3;
        --accent: #f5a524;
        --accent-2: #4fd1c5;
        --danger: #f56565;
        --border: rgba(255, 255, 255, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(245, 165, 36, 0.14), transparent 24%),
            radial-gradient(circle at top right, rgba(79, 209, 197, 0.12), transparent 22%),
            linear-gradient(180deg, #081018 0%, #0a0f14 100%);
        color: var(--text);
    }

    .hero {
        padding: 1.4rem 1.5rem;
        border: 1px solid var(--border);
        border-radius: 1.4rem;
        background: linear-gradient(135deg, rgba(17, 24, 34, 0.96), rgba(23, 34, 48, 0.88));
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }

    .hero-kicker {
        color: var(--accent-2);
        text-transform: uppercase;
        letter-spacing: 0.24em;
        font-size: 0.74rem;
        margin-bottom: 0.35rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
        line-height: 1.02;
        color: var(--text);
    }

    .hero p {
        margin: 0.55rem 0 0;
        color: var(--muted);
        font-size: 0.98rem;
        max-width: 68ch;
    }

    .section-card {
        background: rgba(17, 24, 34, 0.78);
        border: 1px solid var(--border);
        border-radius: 1rem;
        padding: 1rem 1rem 0.75rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(23, 34, 48, 0.96), rgba(17, 24, 34, 0.96));
        border: 1px solid var(--border);
        border-radius: 1rem;
        padding: 0.9rem 1rem;
        margin-bottom: 0.65rem;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.4rem;
        font-weight: 700;
    }

    .metric-hint {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.15rem;
    }
</style>
"""


def load_artifacts() -> tuple[dict[str, Any] | None, pd.DataFrame | None, dict[str, Any] | None]:
    if not RESULTS_PATH.exists() or not ROWS_PATH.exists():
        return None, None, None

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    rows_df = pd.read_csv(ROWS_PATH)
    manifest_path = results.get("manifest_path")
    manifest = None
    if isinstance(manifest_path, str):
        candidate = ROOT / manifest_path
        if candidate.exists():
            manifest = json.loads(candidate.read_text(encoding="utf-8"))

    return results, rows_df, manifest


def format_health_status(state: dict[str, Any] | None) -> str:
    if not isinstance(state, dict):
        return "healthy"
    status = str(state.get("status", "healthy")).strip().lower() or "healthy"
    return status


def build_summary_cards(results: dict[str, Any], rows_df: pd.DataFrame) -> dict[str, Any]:
    outcomes = results.get("outcomes", {}) if isinstance(results.get("outcomes"), dict) else {}
    models = results.get("models", {}) if isinstance(results.get("models"), dict) else {}

    successful_evaluations = int(outcomes.get("success_count", len(rows_df)))
    failure_count = int(results.get("failure_count", 0))
    total_evaluations = successful_evaluations + failure_count
    success_rate = float(outcomes.get("success_rate", 100.0 if total_evaluations else 0.0))
    top_model = "n/a"
    top_score = -1.0
    for model_name, payload in models.items():
        if not isinstance(payload, dict):
            continue
        score = float((payload.get("improved", {}) or {}).get("faithfulness", 0.0))
        if score > top_score:
            top_score = score
            top_model = model_name

    avg_latency = float(rows_df["latency_ms"].mean()) if not rows_df.empty and "latency_ms" in rows_df else 0.0

    return {
        "run_id": results.get("run_id", "unknown"),
        "successful_evaluations": successful_evaluations,
        "failure_count": failure_count,
        "total_evaluations": total_evaluations,
        "success_rate": success_rate,
        "top_model": top_model,
        "avg_latency_ms": avg_latency,
    }


def build_provider_health_frame(results: dict[str, Any]) -> pd.DataFrame:
    provider_health = results.get("provider_health", {}) if isinstance(results.get("provider_health"), dict) else {}
    if not provider_health:
        return pd.DataFrame(
            columns=["provider", "status", "reason", "disabled_until", "last_error", "transient_error_count"]
        )

    rows = []
    for provider, state in provider_health.items():
        if not isinstance(state, dict):
            continue
        rows.append(
            {
                "provider": provider,
                "status": format_health_status(state),
                "reason": state.get("reason", ""),
                "disabled_until": state.get("disabled_until", ""),
                "last_error": state.get("last_error", ""),
                "transient_error_count": state.get("transient_error_count", 0),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["status", "provider"], ascending=[True, True])
    return frame


def build_delta_frame(results: dict[str, Any]) -> pd.DataFrame:
    models = results.get("models", {}) if isinstance(results.get("models"), dict) else {}
    rows = []
    for model_name, payload in models.items():
        if not isinstance(payload, dict):
            continue
        baseline = payload.get("baseline", {}) if isinstance(payload.get("baseline"), dict) else {}
        improved = payload.get("improved", {}) if isinstance(payload.get("improved"), dict) else {}
        delta = payload.get("delta", {}) if isinstance(payload.get("delta"), dict) else {}
        rows.append(
            {
                "model": model_name,
                "baseline_faithfulness": float(baseline.get("faithfulness", 0.0)),
                "improved_faithfulness": float(improved.get("faithfulness", 0.0)),
                "faithfulness_delta": float(delta.get("faithfulness", 0.0)) if delta.get("faithfulness") is not None else None,
                "baseline_hallucination": float(baseline.get("hallucination_rate", 0.0)),
                "improved_hallucination": float(improved.get("hallucination_rate", 0.0)),
                "hallucination_delta": float(delta.get("hallucination_rate", 0.0)) if delta.get("hallucination_rate") is not None else None,
                "baseline_safety": float(baseline.get("safety_score", 0.0)),
                "improved_safety": float(improved.get("safety_score", 0.0)),
                "safety_delta": float(delta.get("safety_score", 0.0)) if delta.get("safety_score") is not None else None,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and "hallucination_delta" in frame:
        frame = frame.sort_values(["hallucination_delta", "faithfulness_delta"], ascending=[True, False])
    return frame


def build_prompt_comparison_frame(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(
            columns=["model", "prompt_version", "answer_relevancy", "faithfulness", "hallucination_rate", "safety_score", "latency_ms"]
        )

    frame = (
        rows_df.groupby(["model", "prompt_version"], as_index=False)
        .agg(
            answer_relevancy=("answer_relevancy", "mean"),
            faithfulness=("faithfulness", "mean"),
            hallucination_rate=("hallucination_rate", "mean"),
            safety_score=("safety_score", "mean"),
            latency_ms=("latency_ms", "mean"),
        )
    )
    return frame.sort_values(["model", "prompt_version"])


def build_failure_frame(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(columns=["model", "prompt_version", "failure_type", "count"])

    frame = (
        rows_df.groupby(["model", "prompt_version", "failure_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "model", "prompt_version"], ascending=[False, True, True])
    )
    return frame


def build_artifact_rows(results: dict[str, Any], manifest: dict[str, Any] | None) -> list[tuple[str, str]]:
    rows = [
        ("results.json", str(RESULTS_PATH.relative_to(ROOT))),
        ("row_results.csv", str(ROWS_PATH.relative_to(ROOT))),
        ("manifest", str(results.get("manifest_path", "n/a"))),
        ("report", str(results.get("report_path", "n/a"))),
    ]
    if isinstance(manifest, dict):
        rows.append(("elapsed_seconds", f"{manifest.get('elapsed_seconds', 'n/a')}"))
        rows.append(("started_at", str(manifest.get("started_at", "n/a"))))
        rows.append(("finished_at", str(manifest.get("finished_at", "n/a"))))
    return rows


def _render_metric(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="LLM Reliability Engine", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    results, rows_df, manifest = load_artifacts()
    if results is None or rows_df is None:
        st.warning("Run `python main.py run-pipeline --live` first to generate artifacts.")
        st.stop()

    summary = build_summary_cards(results, rows_df)
    health_frame = build_provider_health_frame(results)
    delta_frame = build_delta_frame(results)
    comparison_frame = build_prompt_comparison_frame(rows_df)
    failure_frame = build_failure_frame(rows_df)
    artifact_rows = build_artifact_rows(results, manifest)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Operational control room</div>
            <h1>LLM Reliability Engine</h1>
            <p>Financial prompt evaluations, provider health, and prompt deltas in a single live dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_mid, top_right, top_far = st.columns(4)
    with top_left:
        _render_metric("Run ID", str(summary["run_id"]), "Current execution")
    with top_mid:
        _render_metric("Success Rate", f"{summary['success_rate']:.1f}%", f"{summary['successful_evaluations']} successful evaluations")
    with top_right:
        _render_metric("Failures", str(summary["failure_count"]), f"{summary['total_evaluations']} total evaluations")
    with top_far:
        _render_metric("Avg Latency", f"{summary['avg_latency_ms']:.1f} ms", f"Top model: {summary['top_model']}")

    st.caption("Artifacts are sourced from the latest pipeline run. Manifest and report pointers are included for reproducibility.")

    overview_tab, prompts_tab, health_tab, artifacts_tab = st.tabs(["Overview", "Prompts", "Health", "Artifacts"])

    with overview_tab:
        left, right = st.columns([1.15, 0.85])
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Prompt Delta View")
            if delta_frame.empty:
                st.info("No model delta data available yet.")
            else:
                st.dataframe(delta_frame, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Prompt Comparison")
            if comparison_frame.empty:
                st.info("No prompt comparison data available yet.")
            else:
                st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Hallucination by Model/Prompt")
            if not comparison_frame.empty:
                st.bar_chart(comparison_frame.pivot(index="model", columns="prompt_version", values="hallucination_rate"))
            st.markdown('</div>', unsafe_allow_html=True)
        with chart_right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Faithfulness by Model/Prompt")
            if not comparison_frame.empty:
                st.bar_chart(comparison_frame.pivot(index="model", columns="prompt_version", values="faithfulness"))
            st.markdown('</div>', unsafe_allow_html=True)

    with prompts_tab:
        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Model-Prompt Metrics")
            if comparison_frame.empty:
                st.info("No rows to display.")
            else:
                st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Failure Breakdown")
            if failure_frame.empty:
                st.info("No failures captured in the latest run.")
            else:
                st.dataframe(failure_frame, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with health_tab:
        left, right = st.columns([0.9, 1.1])
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Provider Health Snapshot")
            if health_frame.empty:
                st.success("No active provider incidents were recorded in the latest run.")
            else:
                st.dataframe(health_frame, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Run Summary")
            if manifest:
                st.json(
                    {
                        "run_id": summary["run_id"],
                        "elapsed_seconds": manifest.get("elapsed_seconds"),
                        "prompt_versions": manifest.get("runtime", {}).get("prompt_versions", []),
                        "enabled_providers": manifest.get("providers", {}).get("enabled_at_end", []),
                        "disabled_providers": manifest.get("providers", {}).get("disabled_during_run", []),
                    }
                )
            else:
                st.json(results)
            st.markdown('</div>', unsafe_allow_html=True)

    with artifacts_tab:
        left, right = st.columns([0.9, 1.1])
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Artifact Index")
            st.dataframe(pd.DataFrame(artifact_rows, columns=["artifact", "value"]), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Latest Report Preview")
            report_path = RESULTS_PATH.parent / "reports" / f"{results.get('run_id', 'unknown')}.report.md"
            if report_path.exists():
                st.code(report_path.read_text(encoding="utf-8"), language="markdown")
            else:
                st.info("No report file found for the current run.")
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
