"""Data Profiler Agent (role: Research).

Compares schema, missing-value rates, sample size and summary statistics
between the reference data and the production batch. Sets `data_ok`.
"""


def profile(state):
    raise NotImplementedError
