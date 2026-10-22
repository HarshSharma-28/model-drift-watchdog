"""Validation suite: checks the agents against the known, injected ground truth.

Run from the command line:
    python validation.py
or from the "Validation" tab in the Streamlit app.
"""

from __future__ import annotations

import time

import pandas as pd

from data_gen import SCENARIOS
from policy import MAX_MISSING_RATE, P_VALUE_ALPHA, PSI_MODERATE
from runner import run_scenario


def detected_features(result: dict) -> set[str]:
    """Features the agents flagged: significant test or PSI >= 0.10, or a data-quality issue."""
    if not result["quality_report"]["data_ok"]:
        return {f"{c} (missing values)" for c, r in result["quality_report"]["missing_rates"].items() if r > MAX_MISSING_RATE}
    return {r["feature"] for r in result["drift_table"] if r["p_value"] < P_VALUE_ALPHA or r["psi"] >= PSI_MODERATE}


def scenario_runs(use_llm: bool = False) -> pd.DataFrame:
    rows = []
    for labels in (True, False):
        for name, spec in SCENARIOS.items():
            t0 = time.perf_counter()
            r = run_scenario(name, labels_available=labels, use_llm=use_llm)
            secs = time.perf_counter() - t0
            injected, flagged = set(spec.injected), detected_features(r)
            tp = len(injected & flagged)
            rows.append({
                "Scenario": name,
                "Labels": "yes" if labels else "no",
                "Expected": spec.expected,
                "Agent decision": r["decision"],
                "Match": r["decision"] == spec.expected,
                "Critic retries": r["retry_count"],
                "Injected features": ", ".join(sorted(injected)) or "—",
                "Flagged features": ", ".join(sorted(flagged)) or "—",
                "Precision": (tp / len(flagged)) if flagged else (1.0 if not injected else 0.0),
                "Recall": (tp / len(injected)) if injected else 1.0,
                "AUC gain (retrain)": (round(r["model_comparison"]["auc_gain"], 4) if r.get("model_comparison") else None),
                "Seconds": round(secs, 2),
            })
    return pd.DataFrame(rows)


def robustness_runs(use_llm: bool = False) -> pd.DataFrame:
    rows = []
    for name, spec in SCENARIOS.items():
        r = run_scenario(name, inject_faulty_decision=True, use_llm=use_llm)
        first = r["critic_history"][0]
        rows.append({
            "Test": f"Faulty first decision — {name}",
            "What happened": f"Decision agent first said '{first['decision']}'; Critic → {first['verdict']} "
                             f"({first['target']}); final '{r['decision']}'",
            "Expected": spec.expected,
            "Passed": first["verdict"] == "retry" and r["decision"] == spec.expected and not r["escalated"],
        })
    r = run_scenario("Normal week", batch_size=300, use_llm=use_llm)
    rows.append({
        "Test": "Small batch (300 rows)",
        "What happened": f"Critic asked for more data; window widened to {r['window_size']} rows; final '{r['decision']}'",
        "Expected": "Wider window, then No action",
        "Passed": r["window_size"] >= 500 and r["decision"] == "No action" and not r["escalated"],
    })
    r = run_scenario("Sudden drift", batch_size=100, use_llm=use_llm)
    rows.append({
        "Test": "Tiny batch (100 rows)",
        "What happened": f"Window reached {r['window_size']} rows after {r['retry_count']} retries; escalated={r['escalated']}",
        "Expected": "Escalate to human after 2 retries",
        "Passed": r["escalated"] and r["retry_count"] == 2 and r.get("model_comparison") is None,
    })
    a = run_scenario("Gradual drift", use_llm=use_llm, engine="langgraph")
    b = run_scenario("Gradual drift", use_llm=use_llm, engine="fallback")
    rows.append({
        "Test": "LangGraph vs fallback runner",
        "What happened": f"LangGraph '{a['decision']}' / fallback '{b['decision']}'; same path: {a['path'] == b['path']}",
        "Expected": "Identical results",
        "Passed": a["decision"] == b["decision"] and a["path"] == b["path"],
    })
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    s = scenario_runs()
    print("\n=== Scenario ground truth ===")
    print(s[["Scenario", "Labels", "Expected", "Agent decision", "Match", "Critic retries",
             "Precision", "Recall", "AUC gain (retrain)", "Seconds"]].to_string(index=False))
    print(f"\nDecision accuracy: {s['Match'].sum()}/{len(s)}")
    r = robustness_runs()
    print("\n=== Robustness tests ===")
    print(r.to_string(index=False))
    print(f"\nRobustness tests passed: {r['Passed'].sum()}/{len(r)}")


if __name__ == "__main__":
    main()
