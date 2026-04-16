from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "experiments" / "results.json"
ROWS_PATH = ROOT / "experiments" / "row_results.csv"

st.set_page_config(page_title="LLM Reliability Engine", layout="wide")
st.title("LLM Reliability Engine Dashboard")

if not RESULTS_PATH.exists() or not ROWS_PATH.exists():
    st.warning("Run `python main.py run-pipeline --live` first to generate artifacts.")
    st.stop()

results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
rows_df = pd.read_csv(ROWS_PATH)

st.subheader("Run Overview")
st.json(results)

st.subheader("Model-Prompt Metrics")
agg = (
    rows_df.groupby(["model", "prompt_version"], as_index=False)
    .agg(
        answer_relevancy=("answer_relevancy", "mean"),
        faithfulness=("faithfulness", "mean"),
        hallucination_rate=("hallucination_rate", "mean"),
        safety_score=("safety_score", "mean"),
        latency_ms=("latency_ms", "mean"),
    )
)
st.dataframe(agg)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Hallucination by Model/Prompt")
    st.bar_chart(agg.pivot(index="model", columns="prompt_version", values="hallucination_rate"))
with col2:
    st.subheader("Faithfulness by Model/Prompt")
    st.bar_chart(agg.pivot(index="model", columns="prompt_version", values="faithfulness"))

st.subheader("Failure Breakdown")
failures = (
    rows_df.groupby(["model", "prompt_version", "failure_type"], as_index=False)
    .size()
    .rename(columns={"size": "count"})
)
st.dataframe(failures)
