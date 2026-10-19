"""Decision Agent (role: Business Automation).

Applies the locked policy in `policy.py` and returns one of:
No action, Alert, Retrain, Fix pipeline - plus the rule that fired.
"""


def decide(state):
    raise NotImplementedError
