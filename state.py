"""Shared state passed between all agents in the LangGraph workflow.

Every agent reads what it needs from this one object and returns only the
keys it changes. LangGraph merges those updates; the fallback runner does
the same with a plain dict update.
"""

from __future__ import annotations

import time
from typing import Any, Optional, TypedDict

import pandas as pd


class WatchdogState(TypedDict, total=False):
    # ---- Inputs (set by the UI) -------------------------------------------
    scenario: str
    reference_df: pd.DataFrame          # training data
    reference_test_df: pd.DataFrame     # held-out reference rows (baseline)
    baseline_metrics: dict[str, float]  # champion performance at deployment
    champion: Any                       # deployed model (sklearn Pipeline)
    production_pool: pd.DataFrame       # recent production data, newest first
    batch_size: int                     # rows in the monitored batch
    labels_available: bool              # do we already know true outcomes?
    use_llm: bool                       # try Ollama for explanations
    inject_faulty_decision: bool        # demo: Decision agent makes a naive first call
    use_memory: bool                    # read/write run history (agent memory)

    # ---- Planner -----------------------------------------------------------
    plan: list[dict[str, str]]
    replanned: bool

    # ---- Data Profiler -----------------------------------------------------
    window_size: int
    production_df: pd.DataFrame
    quality_report: dict[str, Any]

    # ---- Drift Detector ----------------------------------------------------
    drift_table: list[dict[str, Any]]
    prediction_drift: dict[str, Any]
    performance: Optional[dict[str, Any]]
    bootstrap_features: list[str]

    # ---- Decision Agent ----------------------------------------------------
    decision: str
    decision_rule: str
    decision_evidence: list[str]
    decision_round: int

    # ---- Critic Agent ------------------------------------------------------
    critic_verdict: str                 # "approve" | "retry" | "escalate"
    critic_checks: list[dict[str, Any]]
    critic_history: list[dict[str, Any]]
    retry_count: int
    retry_target: Optional[str]         # "profiler" | "detector" | "decision"
    escalated: bool

    # ---- Reporter / Action Agent -------------------------------------------
    report: str
    explanation: str
    explanation_source: str
    alert: dict[str, str]
    incident: Optional[dict[str, str]]
    model_comparison: Optional[dict[str, Any]]
    challenger: Any

    # ---- Knowledge base & memory -------------------------------------------
    kb_refs: list[dict[str, Any]]       # documents retrieved and cited by agents
    memory_hits: list[dict[str, Any]]   # similar past runs recalled by the Reporter
    runbook_id: Optional[str]           # runbook followed for the final action

    # ---- Bookkeeping -------------------------------------------------------
    trace: list[dict[str, Any]]
    path: list[str]
    started_at: float


def initial_state(**inputs: Any) -> WatchdogState:
    """Fresh state for one watchdog run."""
    state: WatchdogState = {
        "batch_size": 1000,
        "labels_available": True,
        "use_llm": True,
        "inject_faulty_decision": False,
        "use_memory": False,
        "kb_refs": [],
        "memory_hits": [],
        "replanned": False,
        "bootstrap_features": [],
        "decision_round": 0,
        "critic_history": [],
        "retry_count": 0,
        "retry_target": None,
        "escalated": False,
        "performance": None,
        "incident": None,
        "model_comparison": None,
        "challenger": None,
        "trace": [],
        "path": [],
        "started_at": time.perf_counter(),
    }
    state.update(inputs)  # type: ignore[typeddict-item]
    return state
