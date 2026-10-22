"""End-to-end tests: every scenario must reach its ground-truth decision,
and the Critic loop must behave as designed."""

import numpy as np
import pytest

from agents.detector import psi, severity
from data_gen import SCENARIOS
from graph import LANGGRAPH_AVAILABLE
from policy import FIX_PIPELINE, MAX_RETRIES, RETRAIN
from runner import run_scenario


@pytest.mark.parametrize("labels", [True, False])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenario_reaches_expected_decision(scenario, labels):
    r = run_scenario(scenario, labels_available=labels, use_llm=False)
    assert r["decision"] == SCENARIOS[scenario].expected
    assert r["critic_verdict"] == "approve"
    assert not r["escalated"]
    assert r["trace"][0]["agent"] == "Planner"
    assert r["trace"][-1]["agent"] == "Reporter / Action"


def test_pipeline_bug_replans_and_never_retrains():
    r = run_scenario("Pipeline bug", use_llm=False)
    assert "replanner" in r["path"] and "detector" not in r["path"]
    assert r["decision"] == FIX_PIPELINE
    assert r["model_comparison"] is None
    assert r["incident"] is not None


def test_retrain_challenger_beats_champion():
    r = run_scenario("Sudden drift", use_llm=False)
    comp = r["model_comparison"]
    assert r["decision"] == RETRAIN
    assert comp["eligible"] and comp["auc_gain"] >= comp["min_auc_gain"]


def test_gradual_drift_triggers_bootstrap_recheck():
    r = run_scenario("Gradual drift", use_llm=False)
    assert r["critic_history"][0]["target"] == "detector"
    assert any(row["bootstrap"] for row in r["drift_table"])


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_critic_catches_faulty_decision(scenario):
    r = run_scenario(scenario, inject_faulty_decision=True, use_llm=False)
    assert r["critic_history"][0]["verdict"] == "retry"
    assert r["decision"] == SCENARIOS[scenario].expected


def test_small_batch_widens_window():
    r = run_scenario("Normal week", batch_size=300, use_llm=False)
    assert r["window_size"] == 600
    assert r["critic_history"][0]["target"] == "profiler"
    assert r["decision"] == "No action"


def test_tiny_batch_escalates_after_max_retries():
    r = run_scenario("Sudden drift", batch_size=100, use_llm=False)
    assert r["escalated"] and r["retry_count"] == MAX_RETRIES
    assert r["model_comparison"] is None  # nothing acted on automatically


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_langgraph_and_fallback_agree(scenario):
    a = run_scenario(scenario, use_llm=False, engine="langgraph")
    b = run_scenario(scenario, use_llm=False, engine="fallback")
    assert a["decision"] == b["decision"] and a["path"] == b["path"]


def test_psi_basics():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert psi(x, rng.normal(0, 1, 2000)) < 0.10
    assert psi(x, rng.normal(1.0, 1, 2000)) > 0.25
    assert severity(0.05) == "stable" and severity(0.15) == "moderate" and severity(0.4) == "major"
