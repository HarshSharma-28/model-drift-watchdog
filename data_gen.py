"""Synthetic loan-applicant data and the four drift scenarios.

Planned:
- reference data: 5,000 rows used to train the champion model
- production batch: 1,000 rows per scenario
  1. Normal week   - no change              -> expected: No action
  2. Gradual drift - incomes +~20%, city tier shift -> expected: Alert
  3. Sudden drift  - gig-worker influx, lower credit scores -> expected: Retrain
  4. Pipeline bug  - 40% credit_score missing -> expected: Fix pipeline
"""

SCENARIOS = {
    "Normal week": "No action",
    "Gradual drift": "Alert",
    "Sudden drift": "Retrain",
    "Pipeline bug": "Fix pipeline",
}


def generate_reference(n_rows: int = 5000, seed: int = 42):
    """Generate the reference (training) dataset."""
    raise NotImplementedError


def generate_production(scenario: str, n_rows: int = 1000, seed: int = 7):
    """Generate a production batch with the scenario's drift injected."""
    raise NotImplementedError
