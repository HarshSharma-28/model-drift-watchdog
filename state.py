"""Shared state passed between all agents in the LangGraph workflow."""

from typing import Any, Optional, TypedDict

import pandas as pd


class WatchdogState(TypedDict, total=False):
    # Inputs (set by the UI)
    scenario: str
    reference_df: pd.DataFrame
    production_df: pd.DataFrame
    labels_available: bool

    # Planner
    plan: list[str]

    # Data Profiler
    quality_report: dict[str, Any]

    # Drift Detector
    drift_results: dict[str, Any]

    # Decision Agent
    decision: str
    decision_rule: str

    # Critic Agent
    critic_verdict: str            # "approve" | "retry" | "escalate"
    critic_checks: list[dict[str, Any]]
    retry_count: int
    retry_target: Optional[str]    # "profiler" | "detector" | "decision"

    # Reporter / Action Agent
    report: str
    alert: str
    model_comparison: dict[str, Any]

    # Step-by-step log shown in the UI
    trace: list[dict[str, Any]]
