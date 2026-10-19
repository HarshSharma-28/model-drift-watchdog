"""LangGraph orchestration for Model Drift Watchdog.

Planned graph:
    planner -> profiler -> (data bad?  planner re-plans -> decision)
                        -> (data ok)   detector -> decision
    decision -> critic -> approve  -> reporter -> human approval
                       -> retry    -> profiler | detector | decision (max 2)
                       -> escalate -> human

A plain-Python fallback runner executes the same steps if LangGraph
is not installed, so the demo never depends on one library.
"""


def build_graph():
    """Build and compile the LangGraph state graph."""
    raise NotImplementedError


def run_watchdog(state):
    """Run the full workflow and return the final state (with trace)."""
    raise NotImplementedError
