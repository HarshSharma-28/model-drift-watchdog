"""Planner Agent (role: Planner).

Looks at the data before anything runs and builds a task plan:
  * KS test + PSI for each numeric feature
  * Chi-square test + PSI for each categorical feature
  * prediction drift on the champion's scores
  * a performance check, only when true labels are available
Re-plans when the Data Profiler reports a data-quality issue: drift tests on
corrupted data would be misleading, so they are skipped and the run is
routed straight to a Fix-pipeline decision.
"""

from __future__ import annotations

import pandas as pd

from agents.common import log
from policy import TARGET


def _feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = [c for c in df.columns if c != TARGET]
    numeric, categorical = [], []
    for c in features:
        # city_tier is stored as an int but is a category (1/2/3): few distinct values
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 10:
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def plan(state: dict) -> dict:
    reference = state["reference_df"]
    numeric, categorical = _feature_types(reference)
    labels = state.get("labels_available", False)

    tasks = [{"agent": "Data Profiler", "task": "Check schema, missing values, sample size and statistics", "status": "planned"}]
    for f in numeric:
        tasks.append({"agent": "Drift Detector", "task": f"KS test + PSI on `{f}` (numeric)", "status": "planned"})
    for f in categorical:
        tasks.append({"agent": "Drift Detector", "task": f"Chi-square test + PSI on `{f}` (categorical)", "status": "planned"})
    tasks.append({"agent": "Drift Detector", "task": "Prediction drift: PSI on champion's risk scores", "status": "planned"})
    if labels:
        tasks.append({"agent": "Drift Detector", "task": "Performance check: accuracy & AUC vs deployment baseline", "status": "planned"})
    else:
        tasks.append({
            "agent": "Drift Detector",
            "task": "Performance check skipped: loan outcomes not known yet (label delay)",
            "status": "skipped",
        })
    tasks += [
        {"agent": "Decision Agent", "task": "Apply decision policy", "status": "planned"},
        {"agent": "Critic Agent", "task": "Validate evidence and decision (max 2 retries)", "status": "planned"},
        {"agent": "Reporter / Action", "task": "Report, alert, and act (retrain if approved)", "status": "planned"},
    ]

    reasoning = (
        f"Found {len(numeric)} numeric features ({', '.join(numeric)}) → KS + PSI; "
        f"{len(categorical)} categorical ({', '.join(categorical)}) → Chi-square + PSI. "
        + ("Labels are available, so a performance check is added." if labels
           else "No labels yet, so monitoring relies on input and prediction drift only.")
    )
    return {
        "plan": tasks,
        "trace": log(state, "Planner", "Built task plan", reasoning, "ok",
                     {"numeric_features": numeric, "categorical_features": categorical,
                      "labels_available": labels, "tasks": len(tasks)}),
        "path": state.get("path", []) + ["planner"],
    }


def replan(state: dict) -> dict:
    """Called when the Profiler finds bad data."""
    issues = state["quality_report"]["issues"]
    new_plan = []
    for t in state["plan"]:
        t = dict(t)
        if t["agent"] == "Drift Detector":
            t["status"] = "skipped"
            t["task"] = t["task"] + " — skipped (data-quality issue)"
        new_plan.append(t)
    new_plan.insert(1, {"agent": "Planner", "task": "Re-plan: route to Fix-pipeline decision", "status": "done"})
    return {
        "plan": new_plan,
        "replanned": True,
        "trace": log(
            state, "Planner", "Re-planned",
            "Data-quality issue found (" + "; ".join(issues) + "). Drift tests on corrupted data "
            "would be misleading, so they are skipped and the run goes straight to the Decision agent.",
            "warn", {"issues": issues},
        ),
        "path": state.get("path", []) + ["replanner"],
    }
