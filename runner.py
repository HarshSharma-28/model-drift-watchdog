"""Convenience layer: build the inputs for one watchdog run and execute it.

Used by the Streamlit app, the validation script and the tests.
"""

from __future__ import annotations

from functools import lru_cache

from data_gen import SCENARIOS, generate_pool, generate_reference, train_champion
from graph import run_watchdog
from state import initial_state


@lru_cache(maxsize=1)
def load_context():
    """Reference data + the deployed champion model (built once)."""
    reference = generate_reference()
    champion, reference_test, baseline = train_champion(reference)
    return reference, champion, reference_test, baseline


@lru_cache(maxsize=8)
def load_pool(scenario: str):
    return generate_pool(scenario)


def run_scenario(
    scenario: str,
    labels_available: bool = True,
    batch_size: int = 1000,
    use_llm: bool = True,
    inject_faulty_decision: bool = False,
    use_memory: bool = False,
    engine: str = "auto",
    champion=None,
    baseline: dict | None = None,
) -> dict:
    """Run one monitoring cycle. `champion`/`baseline` override the deployed model
    (used after a human approves a retrained challenger in the UI)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    reference, default_champion, reference_test, default_baseline = load_context()
    state = initial_state(
        scenario=scenario,
        reference_df=reference,
        reference_test_df=reference_test,
        baseline_metrics=baseline or default_baseline,
        champion=champion or default_champion,
        production_pool=load_pool(scenario),
        batch_size=batch_size,
        labels_available=labels_available,
        use_llm=use_llm,
        inject_faulty_decision=inject_faulty_decision,
        use_memory=use_memory,
    )
    return run_watchdog(state, engine=engine)
