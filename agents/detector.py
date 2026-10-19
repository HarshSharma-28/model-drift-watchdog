"""Drift Detector Agent (role: Executor).

Runs KS (numeric), Chi-square (categorical) and PSI (all features),
prediction drift on the champion model, and accuracy/AUC when labels
exist. Supports a bootstrap re-check requested by the Critic.
"""


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two samples."""
    raise NotImplementedError


def detect(state):
    raise NotImplementedError
