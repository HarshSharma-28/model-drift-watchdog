"""Knowledge base retrieval, agent citations and memory."""

import pytest

import knowledge
from data_gen import SCENARIOS
from knowledge import get_kb, recall, remember
from runner import run_scenario

EXPECTED_RUNBOOK = {"Normal week": "RB-01", "Gradual drift": "RB-02", "Sudden drift": "RB-03", "Pipeline bug": "RB-04"}


def test_kb_loads_all_categories():
    kb = get_kb()
    assert len(kb) >= 20
    assert {d.category for d in kb.docs} >= {"concept", "policy", "runbook", "domain"}
    assert all(get_kb().get(rb).steps for rb in ["RB-01", "RB-02", "RB-03", "RB-04", "RB-05"])


@pytest.mark.parametrize("query,expected", [
    ("KS test for numeric feature drift", "C-03"),
    ("PSI population stability index severity", "C-02"),
    ("missing values broken data pipeline incident", "RB-04"),
    ("critic validation retry escalation rules", "P-03"),
    ("gig workers thin credit file", "D-04"),
    ("label delay outcomes months later", "D-02"),
])
def test_retrieval_top_hit(query, expected):
    assert get_kb().search(query, k=1)[0][0].id == expected


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_agents_cite_kb_and_follow_runbook(scenario):
    r = run_scenario(scenario, use_llm=False)
    ids = {x["id"] for x in r["kb_refs"]}
    assert r["runbook_id"] == EXPECTED_RUNBOOK[scenario]
    assert {"P-01", "P-03", r["runbook_id"]} <= ids
    assert f"Runbook {r['runbook_id']}" in r["report"]
    assert any(e["kb"] for e in r["trace"])


def test_escalation_uses_escalation_runbook():
    r = run_scenario("Sudden drift", batch_size=100, use_llm=False)
    assert r["escalated"] and r["runbook_id"] == "RB-05"


def test_memory_recalls_similar_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "MEMORY_PATH", tmp_path / "memory.jsonl")
    assert recall("Retrain", ["credit_score"]) == []
    remember({"scenario": "Sudden drift", "decision": "Retrain", "drifted": ["credit_score", "employment_type"]})
    remember({"scenario": "Normal week", "decision": "No action", "drifted": []})
    hits = recall("Retrain", ["credit_score", "employment_type"])
    assert len(hits) == 1 and hits[0]["scenario"] == "Sudden drift"


def test_memory_in_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "MEMORY_PATH", tmp_path / "memory.jsonl")
    first = run_scenario("Pipeline bug", use_llm=False, use_memory=True)
    second = run_scenario("Pipeline bug", use_llm=False, use_memory=True)
    assert first["memory_hits"] == [] and len(second["memory_hits"]) == 1
    assert first["decision"] == second["decision"] == "Fix pipeline"  # memory never changes decisions
