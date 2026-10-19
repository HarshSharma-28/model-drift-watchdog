"""Planner Agent (role: Planner).

Builds the task plan from the data: chooses KS for numeric features,
Chi-square for categorical, PSI for all, and adds a performance check
when labels exist. Re-plans to a Fix-pipeline route when the Profiler
reports a data-quality issue.
"""


def plan(state):
    raise NotImplementedError


def replan(state):
    raise NotImplementedError
