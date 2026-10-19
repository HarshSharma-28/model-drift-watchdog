"""Critic Agent (role: Critic).

Checks sample size, KS/PSI agreement on critical features and
decision-policy consistency. Returns approve, retry(target) or escalate.
Maximum 2 retries before escalating to a human.
"""


def review(state):
    raise NotImplementedError
