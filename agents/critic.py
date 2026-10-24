"""Critic Agent (role: Critic).

Validates the evidence and the decision before anything is acted on.

Checks (the first failing one decides where work is sent back):
  1. Sample size >= 500                         -> else retry Profiler (wider window)
  2. KS/Chi² and PSI agree on critical features -> else retry Detector (bootstrap)
  3. Never retrain on corrupted data            -> else retry Decision
  4. Decision follows the policy                -> else retry Decision
     (the Critic re-derives the expected action independently)
  5. Decision is backed by recorded evidence    -> else retry Decision

At most 2 retries; if checks still fail, the run is escalated to a human.
"""

from __future__ import annotations

from agents.common import log
from knowledge import cite
from policy import (ALERT, FIX_PIPELINE, MAX_ACCURACY_DROP, MAX_RETRIES, MIN_SAMPLE_SIZE,
                    NO_ACTION, PSI_MAJOR, PSI_MODERATE, RETRAIN)


def expected_decision(state: dict) -> str:
    """Independent re-derivation of the policy from raw evidence."""
    q = state["quality_report"]
    if not q["data_ok"]:
        return FIX_PIPELINE
    table, pred, perf = state["drift_table"], state["prediction_drift"], state.get("performance")
    if any(r["critical"] and r["severity"] == "major" for r in table):
        return RETRAIN
    if pred["psi"] > PSI_MAJOR or (perf is not None and perf["accuracy_drop"] > MAX_ACCURACY_DROP):
        return RETRAIN
    if any(r["severity"] in ("moderate", "major") for r in table) or pred["psi"] >= PSI_MODERATE:
        return ALERT
    return NO_ACTION


def run_checks(state: dict) -> list[dict]:
    q = state["quality_report"]
    decision = state["decision"]
    detector_ran = q["data_ok"] and "drift_table" in state
    checks = []

    checks.append({
        "check": "Sample size is large enough",
        "passed": q["rows"] >= MIN_SAMPLE_SIZE,
        "detail": f"{q['rows']:,} rows (minimum {MIN_SAMPLE_SIZE})",
        "on_fail": "profiler",
    })

    if detector_ran:
        disputed = [r["feature"] for r in state["drift_table"] if r["critical"] and r["disagree"]]
        checks.append({
            "check": "KS/Chi² and PSI agree on critical features",
            "passed": not disputed,
            "detail": ("All critical features agree" if not disputed
                       else "Disagreement on " + ", ".join(disputed) + " (significant test but small PSI, or vice versa)"),
            "on_fail": "detector",
            "disputed": disputed,
        })
    else:
        checks.append({"check": "KS/Chi² and PSI agree on critical features", "passed": None,
                       "detail": "N/A — drift tests skipped by the Planner (data-quality issue)", "on_fail": "detector"})

    checks.append({
        "check": "Never retrain on corrupted data",
        "passed": not (decision == RETRAIN and not q["data_ok"]),
        "detail": "Data quality OK" if q["data_ok"] else
                  ("Data is corrupted and decision is " + decision),
        "on_fail": "decision",
    })

    expected = expected_decision(state)
    checks.append({
        "check": "Decision follows the policy",
        "passed": decision == expected,
        "detail": f"Decision '{decision}', policy says '{expected}'",
        "on_fail": "decision",
    })

    evidence = state.get("decision_evidence") or []
    checks.append({
        "check": "Decision is backed by evidence",
        "passed": bool(evidence) and evidence != ["no evidence checked"],
        "detail": "; ".join(evidence[:3]) if evidence else "No evidence recorded",
        "on_fail": "decision",
    })
    return checks


def review(state: dict) -> dict:
    checks = run_checks(state)
    failed = [c for c in checks if c["passed"] is False]
    retries = state.get("retry_count", 0)
    history = list(state.get("critic_history", []))

    update: dict = {}
    if not failed:
        verdict, target, status = "approve", None, "ok"
        summary = f"All {sum(c['passed'] is not None for c in checks)} applicable checks passed → approved '{state['decision']}'."
    elif retries < MAX_RETRIES:
        first = failed[0]
        verdict, target, status = "retry", first["on_fail"], "retry"
        summary = f"Failed: {first['check']} ({first['detail']}). Sending back to the {target} (retry {retries + 1}/{MAX_RETRIES})."
        update["retry_count"] = retries + 1
        if target == "detector":
            update["bootstrap_features"] = first.get("disputed", [])
    else:
        verdict, target, status = "escalate", None, "fail"
        summary = (f"Still failing after {MAX_RETRIES} retries ({failed[0]['check']}). "
                   "Escalating to a human — no automatic action will be taken.")
        update["escalated"] = True

    refs, used = state.get("kb_refs", []), []
    refs, i = cite(refs, "Critic Agent", "critic validation retry escalation rules", "Validation rules", category="policy")
    used.append(i)
    if verdict == "escalate":
        refs, i = cite(refs, "Critic Agent", "escalation human review insufficient evidence",
                       "Escalation procedure", category="runbook")
        used.append(i)
    update["kb_refs"] = refs

    history.append({"round": len(history) + 1, "verdict": verdict, "target": target,
                    "decision": state["decision"], "checks": checks})
    update.update({
        "critic_verdict": verdict,
        "critic_checks": checks,
        "critic_history": history,
        "retry_target": target,
        "trace": log(state, "Critic Agent", {"approve": "Approved", "retry": "Sent back for retry",
                                             "escalate": "Escalated to human"}[verdict],
                     summary, status, {"failed_checks": [c["check"] for c in failed], "retry_count": update.get("retry_count", retries)},
                     kb=used),
        "path": state.get("path", []) + ["critic"],
    })
    return update
