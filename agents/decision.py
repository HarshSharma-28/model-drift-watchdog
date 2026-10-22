"""Decision Agent (role: Business Automation).

Applies the locked policy from `policy.py` and records exactly which rule
fired and the evidence behind it:

  Data-quality issue                                         -> Fix pipeline
  Major drift on a critical feature, prediction PSI > 0.25,
  or accuracy drop > 5 points                                -> Retrain
  Moderate drift anywhere, or major on a non-critical feature -> Alert
  Otherwise                                                   -> No action

Demo option: `inject_faulty_decision` makes the *first* call a naive one
(the kind of mistake a simple script makes), so the Critic can be shown
catching it and sending it back.
"""

from __future__ import annotations

from agents.common import fmt_pct, log
from policy import (ALERT, FIX_PIPELINE, MAX_ACCURACY_DROP, NO_ACTION, PSI_MAJOR, RETRAIN)


def apply_policy(state: dict) -> tuple[str, str, list[str]]:
    q = state["quality_report"]
    if not q["data_ok"]:
        return FIX_PIPELINE, "Data-quality issue → never test or retrain on corrupted data", list(q["issues"])

    table = state["drift_table"]
    pred = state["prediction_drift"]
    perf = state.get("performance")

    major_critical = [r for r in table if r["severity"] == "major" and r["critical"]]
    retrain_evidence = [f"`{r['feature']}` PSI {r['psi']:.3f} (major, critical feature)" for r in major_critical]
    if pred["psi"] > PSI_MAJOR:
        retrain_evidence.append(f"prediction PSI {pred['psi']:.3f} > {PSI_MAJOR}")
    if perf and perf["accuracy_drop"] > MAX_ACCURACY_DROP:
        retrain_evidence.append(
            f"accuracy dropped {perf['accuracy_drop'] * 100:.1f} points "
            f"({fmt_pct(perf['baseline_accuracy'])} → {fmt_pct(perf['current_accuracy'])})")
    if retrain_evidence:
        return RETRAIN, "Major drift on a critical feature / prediction drift / accuracy drop", retrain_evidence

    moderate = [r for r in table if r["severity"] == "moderate"]
    major_noncritical = [r for r in table if r["severity"] == "major" and not r["critical"]]
    alert_evidence = [f"`{r['feature']}` PSI {r['psi']:.3f} ({r['severity']})" for r in moderate + major_noncritical]
    if pred["severity"] == "moderate":
        alert_evidence.append(f"prediction PSI {pred['psi']:.3f} (moderate)")
    if alert_evidence:
        return ALERT, "Moderate drift, or major drift on a non-critical feature", alert_evidence

    return NO_ACTION, "All features and predictions stable (PSI < 0.10)", [
        f"max feature PSI {max(r['psi'] for r in table):.3f}", f"prediction PSI {pred['psi']:.3f}"]


def _naive(correct: str) -> tuple[str, str, list[str]]:
    """What a naive script would do: retrain whenever new data arrives."""
    wrong = NO_ACTION if correct == RETRAIN else RETRAIN
    return wrong, "Naive rule (fault injected for demo): " + (
        "new data arrived → retrain" if wrong == RETRAIN else "model still running → no action"), ["no evidence checked"]


def decide(state: dict) -> dict:
    round_no = state.get("decision_round", 0) + 1
    revising = state.get("retry_target") == "decision"
    decision, rule, evidence = apply_policy(state)

    faulty = state.get("inject_faulty_decision") and round_no == 1
    if faulty:
        decision, rule, evidence = _naive(decision)

    if revising:
        action = "Revised decision"
        summary = f"Critic flagged a policy violation. Re-applied the policy strictly → **{decision}** ({rule})."
    else:
        action = "Applied policy"
        summary = f"Decision: **{decision}** — {rule}."
    if faulty:
        summary += " ⚠️ Fault injected for the demo."

    return {
        "decision": decision,
        "decision_rule": rule,
        "decision_evidence": evidence,
        "decision_round": round_no,
        "retry_target": None if revising else state.get("retry_target"),
        "trace": log(state, "Decision Agent", action, summary, "warn" if faulty else "ok",
                     {"decision": decision, "rule": rule, "evidence": evidence, "round": round_no}),
        "path": state.get("path", []) + ["decision"],
    }
