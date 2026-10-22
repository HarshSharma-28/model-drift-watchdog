"""Reporter / Action Agent (roles: Executor + Business Automation).

After the Critic approves (or escalates), this agent:
  * writes a structured drift report and a plain-English explanation
    (local Ollama model if available, otherwise a template-based mock)
  * routes an alert to the right team
  * opens an incident ticket for pipeline problems
  * on Retrain: trains a challenger model on reference + recent labelled data
    and compares it with the champion on fresh, unseen data.
Nothing is deployed here - a human must approve in the UI.
"""

from __future__ import annotations

import time
import zlib

import numpy as np
import pandas as pd

from agents.common import fmt_pct, log
from data_gen import FEATURES, build_model, evaluate, generate_holdout
from llm import explain
from policy import ALERT, FIX_PIPELINE, MIN_AUC_GAIN, NO_ACTION, RETRAIN, TARGET

RECENT_WEIGHT = 3.0   # recent rows count 3x when retraining

ALERT_ROUTING = {
    NO_ACTION: ("ℹ️", "info", "Logged only — no one paged"),
    ALERT: ("⚠️", "warning", "#ml-monitoring channel (ML team)"),
    RETRAIN: ("🔁", "high", "ML team + Credit Risk lead (approval needed)"),
    FIX_PIPELINE: ("🛠️", "high", "Data Engineering on-call"),
}


def retrain_and_compare(state: dict) -> dict:
    reference: pd.DataFrame = state["reference_df"]
    recent: pd.DataFrame = state["production_df"].dropna(subset=FEATURES)
    holdout = generate_holdout(state["scenario"])

    data = pd.concat([reference, recent], ignore_index=True)
    weights = np.r_[np.ones(len(reference)), np.full(len(recent), RECENT_WEIGHT)]
    t0 = time.perf_counter()
    challenger = build_model(random_state=1)
    challenger.fit(data[FEATURES], data[TARGET], clf__sample_weight=weights)
    train_s = time.perf_counter() - t0

    champ = evaluate(state["champion"], holdout)
    chall = evaluate(challenger, holdout)
    gain = chall["auc"] - champ["auc"]
    return {
        "challenger_model": challenger,
        "comparison": {
            "holdout_rows": len(holdout),
            "training_rows": len(data),
            "recent_rows": len(recent),
            "train_seconds": round(train_s, 2),
            "champion": champ,
            "challenger": chall,
            "auc_gain": gain,
            "accuracy_gain": chall["accuracy"] - champ["accuracy"],
            "min_auc_gain": MIN_AUC_GAIN,
            "eligible": gain >= MIN_AUC_GAIN,
        },
    }


def _facts(state: dict) -> str:
    lines = [f"Scenario monitored: {state['scenario']}", f"Rows checked: {state['quality_report']['rows']}"]
    if not state["quality_report"]["data_ok"]:
        lines.append("Data-quality issues: " + "; ".join(state["quality_report"]["issues"]))
    else:
        for r in state["drift_table"]:
            lines.append(f"{r['feature']}: PSI {r['psi']:.3f} ({r['severity']}), {r['test']} p-value {r['p_value']:.3g}")
        lines.append(f"Prediction PSI: {state['prediction_drift']['psi']:.3f}")
        if state.get("performance"):
            p = state["performance"]
            lines.append(f"Accuracy {fmt_pct(p['current_accuracy'])} vs baseline {fmt_pct(p['baseline_accuracy'])}")
    lines.append(f"Decision: {state['decision']} ({state['decision_rule']})")
    lines.append("Critic: " + ("escalated to human" if state.get("escalated") else "approved"))
    return "\n".join(lines)


def _mock_explanation(state: dict, comparison: dict | None) -> str:
    d = state["decision"]
    if state.get("escalated"):
        return ("The monitoring agents could not reach a decision they could fully validate, so nothing was changed "
                "automatically. A person from the ML team should review the evidence below and decide the next step.")
    if d == FIX_PIPELINE:
        issue = state["quality_report"]["issues"][0]
        return (f"This week's applicant data is broken rather than genuinely different: {issue}. "
                "Retraining on this data would teach the model from corrupted inputs, so the agents did not retrain. "
                "Data Engineering has been asked to fix the upstream feed; the check will be re-run once it is restored.")
    drifted = [r for r in state["drift_table"] if r["severity"] != "stable"]
    names = ", ".join(r["feature"].replace("_", " ") for r in drifted) or "no feature"
    if d == NO_ACTION:
        return ("Applicants this week look just like the people the model was trained on. "
                "No input feature or prediction pattern moved meaningfully, so the current model stays in place and nothing needs to be done.")
    if d == ALERT:
        return (f"Some applicant characteristics are shifting ({names}), but not enough yet to harm the model's decisions. "
                "The ML team has been alerted to keep watching; if the shift keeps growing, retraining will be recommended.")
    text = (f"The applicant population has changed sharply ({names}), and the current model is now judging risk on people "
            "it was not trained for. The agents recommend retraining.")
    if comparison:
        c = comparison
        text += (f" A retrained model scored AUC {c['challenger']['auc']:.3f} versus {c['champion']['auc']:.3f} for the current one "
                 f"on fresh data ({'+' if c['auc_gain'] >= 0 else ''}{c['auc_gain']:.3f}). ")
        text += ("It is ready for human approval." if c["eligible"] else
                 "That improvement is below the 0.01 bar, so the current model is kept.")
    return text


def _report_markdown(state: dict, comparison: dict | None) -> str:
    q = state["quality_report"]
    out = [f"#### Drift report — {state['scenario']}",
           f"**Decision:** {state['decision']}" + (" (escalated to human)" if state.get("escalated") else ""),
           f"**Rule:** {state['decision_rule']}",
           "**Evidence:** " + "; ".join(state.get("decision_evidence", [])),
           f"**Data checked:** {q['rows']:,} production rows vs {len(state['reference_df']):,} reference rows"]
    if q["data_ok"]:
        drifted = [r for r in state["drift_table"] if r["severity"] != "stable"]
        out.append("**Drifted features:** " + (", ".join(f"{r['feature']} ({r['severity']}, PSI {r['psi']:.3f})" for r in drifted) or "none"))
        out.append(f"**Prediction drift:** PSI {state['prediction_drift']['psi']:.3f}")
        if state.get("performance"):
            p = state["performance"]
            out.append(f"**Performance:** accuracy {fmt_pct(p['current_accuracy'])} (baseline {fmt_pct(p['baseline_accuracy'])}), "
                       f"AUC {p['current_auc']:.3f} (baseline {p['baseline_auc']:.3f})")
    else:
        out.append("**Data-quality issues:** " + "; ".join(q["issues"]))
    passed = sum(1 for c in state["critic_checks"] if c["passed"])
    applicable = sum(1 for c in state["critic_checks"] if c["passed"] is not None)
    out.append(f"**Critic:** {passed}/{applicable} checks passed after {state.get('retry_count', 0)} retr{'y' if state.get('retry_count', 0) == 1 else 'ies'}")
    if comparison:
        out.append(f"**Champion vs challenger (fresh data):** AUC {comparison['champion']['auc']:.3f} → "
                   f"{comparison['challenger']['auc']:.3f} (gain {comparison['auc_gain']:+.3f}; "
                   f"{'eligible' if comparison['eligible'] else 'not eligible'} for deployment)")
    return "\n\n".join(out)


def report(state: dict) -> dict:
    decision = state["decision"]
    escalated = state.get("escalated", False)

    comparison, challenger = None, None
    if decision == RETRAIN and not escalated:
        result = retrain_and_compare(state)
        comparison, challenger = result["comparison"], result["challenger_model"]

    fallback = _mock_explanation(state, comparison)
    explanation, source = explain(_facts(state), fallback, use_llm=state.get("use_llm", True))

    icon, level, route = ALERT_ROUTING[decision]
    if escalated:
        icon, level, route = "🚨", "high", "ML team lead (manual review)"
    alert = {
        "icon": icon, "level": level, "route": route,
        "title": f"{icon} Model Drift Watchdog — {('ESCALATED: ' if escalated else '')}{decision}",
        "message": f"[{state['scenario']}] {state['decision_rule']}. " + "; ".join(state.get("decision_evidence", [])[:3]),
    }

    incident = None
    if decision == FIX_PIPELINE:
        bad = [c for c, r in state["quality_report"]["missing_rates"].items() if r > 0.10]
        incident = {
            "id": f"INC-{zlib.crc32(state['scenario'].encode()) % 9000 + 1000}",
            "owner": "Data Engineering",
            "title": "Upstream data feed delivering incomplete records",
            "affected": ", ".join(bad) or "schema",
            "action": "Restore the feed, back-fill missing values, then re-run the watchdog. Do NOT retrain until fixed.",
        }

    summary = f"Wrote report and routed alert to {route}."
    if comparison:
        summary += (f" Trained challenger on {comparison['training_rows']:,} rows: AUC {comparison['champion']['auc']:.3f} → "
                    f"{comparison['challenger']['auc']:.3f} on {comparison['holdout_rows']:,} fresh rows "
                    f"({'eligible' if comparison['eligible'] else 'not eligible'} — awaiting human approval).")
    if incident:
        summary += f" Opened incident {incident['id']} for {incident['owner']}."
    summary += f" Explanation by {'Ollama' if source.startswith('ollama') else 'mock LLM'}."

    return {
        "report": _report_markdown(state, comparison),
        "explanation": explanation,
        "explanation_source": source,
        "alert": alert,
        "incident": incident,
        "model_comparison": comparison,
        "challenger": challenger,
        "trace": log(state, "Reporter / Action", "Reported & acted", summary, "ok",
                     {"alert_route": route, "explanation_source": source,
                      "auc_gain": round(comparison["auc_gain"], 4) if comparison else None}),
        "path": state.get("path", []) + ["reporter"],
    }
