"""LangGraph orchestration for Model Drift Watchdog.

    planner ─► profiler ─┬─ data OK ──► detector ─► decision ─► critic ─┬─ approve ──► reporter ─► END
                         └─ bad data ─► replanner ──┘                   ├─ retry ────► profiler | detector | decision
                                                                        └─ escalate ─► reporter (escalated) ─► END

Human approval of a retrained model happens after the graph, in the UI.

All agents share one state object. Routing lives in two small functions
(`route_after_profiler`, `route_after_critic`) used by BOTH the LangGraph
graph and the plain-Python fallback runner, so the two always behave the same.
"""

from __future__ import annotations

from typing import Callable

from agents.critic import review
from agents.decision import decide
from agents.detector import detect
from agents.planner import plan, replan
from agents.profiler import profile
from agents.reporter import report
from state import WatchdogState

try:  # LangGraph is the preferred orchestrator
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when langgraph is missing
    LANGGRAPH_AVAILABLE = False

NODES: dict[str, Callable[[dict], dict]] = {
    "planner": plan,
    "profiler": profile,
    "replanner": replan,
    "detector": detect,
    "decision": decide,
    "critic": review,
    "reporter": report,
}


# ---------------------------------------------------------------------------
# Routing (shared)
# ---------------------------------------------------------------------------

def route_after_profiler(state: dict) -> str:
    return "detector" if state["quality_report"]["data_ok"] else "replanner"


def route_after_critic(state: dict) -> str:
    verdict = state["critic_verdict"]
    if verdict == "retry":
        return state["retry_target"]          # profiler | detector | decision
    return "reporter"                         # approve or escalate


EDGES = {"planner": "profiler", "replanner": "decision", "detector": "decision",
         "decision": "critic", "reporter": None}
CONDITIONAL = {"profiler": route_after_profiler, "critic": route_after_critic}


# ---------------------------------------------------------------------------
# LangGraph
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph state graph."""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph is not installed")
    g = StateGraph(WatchdogState)
    for name, fn in NODES.items():
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "profiler")
    g.add_conditional_edges("profiler", route_after_profiler, {"detector": "detector", "replanner": "replanner"})
    g.add_edge("replanner", "decision")
    g.add_edge("detector", "decision")
    g.add_edge("decision", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"profiler": "profiler", "detector": "detector",
                             "decision": "decision", "reporter": "reporter"})
    g.add_edge("reporter", END)
    return g.compile()


_COMPILED = None


def _graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


# ---------------------------------------------------------------------------
# Fallback runner (same nodes, same routing, no dependency)
# ---------------------------------------------------------------------------

def run_fallback(state: dict, max_steps: int = 50) -> dict:
    state = dict(state)
    node = "planner"
    for _ in range(max_steps):
        state.update(NODES[node](state))
        if node in CONDITIONAL:
            node = CONDITIONAL[node](state)
        else:
            node = EDGES[node]
        if node is None:
            return state
    raise RuntimeError("Workflow did not finish within the step limit")


def run_watchdog(state: dict, engine: str = "auto") -> dict:
    """Run the full workflow and return the final state.

    engine: "langgraph", "fallback" or "auto" (LangGraph when installed).
    """
    use_langgraph = engine == "langgraph" or (engine == "auto" and LANGGRAPH_AVAILABLE)
    if use_langgraph:
        final = _graph().invoke(state, config={"recursion_limit": 50})
        final["engine"] = "LangGraph"
        return final
    final = run_fallback(state)
    final["engine"] = "Fallback runner"
    return final
