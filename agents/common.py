"""Small helpers shared by all agents."""

from __future__ import annotations

import time
from typing import Any

AGENT_META = {
    "Planner": {"icon": "🧭", "role": "Planner"},
    "Data Profiler": {"icon": "🔎", "role": "Research"},
    "Drift Detector": {"icon": "📡", "role": "Executor"},
    "Decision Agent": {"icon": "🔀", "role": "Business Automation"},
    "Critic Agent": {"icon": "🛡️", "role": "Critic"},
    "Reporter / Action": {"icon": "📝", "role": "Executor + Automation"},
    "Human": {"icon": "👤", "role": "Approver"},
}


def log(
    state: dict,
    agent: str,
    action: str,
    summary: str,
    status: str = "ok",
    details: dict[str, Any] | None = None,
    kb: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the trace with one new entry appended (never mutates the input)."""
    trace = list(state.get("trace", []))
    meta = AGENT_META.get(agent, {"icon": "•", "role": ""})
    trace.append(
        {
            "step": len(trace) + 1,
            "agent": agent,
            "icon": meta["icon"],
            "role": meta["role"],
            "action": action,
            "summary": summary,
            "status": status,  # ok | warn | fail | retry | info
            "details": details or {},
            "kb": [k for k in (kb or []) if k],
            "t_ms": round((time.perf_counter() - state.get("started_at", time.perf_counter())) * 1000),
        }
    )
    return trace


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"
