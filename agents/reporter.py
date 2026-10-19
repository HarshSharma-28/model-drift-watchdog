"""Reporter / Action Agent (roles: Executor + Business Automation).

Writes the plain-English report and alert. On Retrain, trains a
challenger model and compares it with the champion on fresh data.
Deployment still requires human approval.
"""


def report(state):
    raise NotImplementedError


def retrain_and_compare(state):
    raise NotImplementedError
