"""Model Drift Watchdog - Streamlit entry point.

Placeholder page while the PoC is being built. Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Model Drift Watchdog", page_icon="🛰️", layout="wide")

st.title("🛰️ Model Drift Watchdog")
st.caption("An Agentic AI system that detects, decides, validates and acts when ML models go stale.")
st.info("🚧 PoC under development. The full agent workflow and demo scenarios are coming soon.")

st.subheader("Team")
st.markdown(
    "- **Harsh Sharma** — PRN 24070126067\n"
    "- **Divyansh Agarwal** — PRN 24070126056"
)

st.subheader("Planned agents")
st.table(
    {
        "Agent": ["Planner", "Data Profiler", "Drift Detector", "Decision", "Critic", "Reporter / Action", "Human"],
        "Responsibility": [
            "Builds the task plan; re-plans on bad data",
            "Schema, missing values, sample size",
            "KS, Chi-square, PSI, prediction drift",
            "No action / Alert / Retrain / Fix pipeline",
            "Validates, retries (max 2) or escalates",
            "Report, alert, champion vs challenger",
            "Approves deployment",
        ],
    }
)
