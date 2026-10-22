"""Drift Detector Agent (role: Executor).

Runs the statistical tests chosen by the Planner:
  * KS test (numeric) / Chi-square test (categorical) -> is the change significant?
  * PSI (all features)                               -> how big is the change?
  * Prediction drift: PSI on the champion's risk scores
  * Performance (only with labels): accuracy & AUC vs the deployment baseline

When the Critic finds that KS and PSI disagree on a critical feature, it asks
the Detector to re-check those features with a bootstrap: PSI is recomputed on
300 resamples of the batch, and the 95% interval settles whether the change is
practically meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from agents.common import log
from data_gen import FEATURES, evaluate
from policy import (CRITICAL_FEATURES, MAX_ACCURACY_DROP, P_VALUE_ALPHA, PSI_MAJOR,
                    PSI_MODERATE, TARGET)

EPS = 1e-4
N_BOOTSTRAP = 300


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 10


def psi(expected, actual, bins: int = 10, categorical: bool = False) -> float:
    """Population Stability Index between a reference sample and a new sample."""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    if categorical:
        cats = sorted(set(expected.unique()) | set(actual.unique()), key=str)
        e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, EPS, None)
    a = np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def severity(psi_value: float) -> str:
    if np.isnan(psi_value):
        return "n/a"
    if psi_value > PSI_MAJOR:
        return "major"
    if psi_value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def tests_disagree(row: dict) -> bool:
    """KS/Chi² says 'significant' but PSI says 'stable', or the other way round."""
    return (row["drifted"] and row["psi"] < PSI_MODERATE) or (not row["drifted"] and row["psi"] >= PSI_MODERATE)


def _test_feature(ref: pd.Series, prod: pd.Series, name: str) -> dict:
    ref, prod = ref.dropna(), prod.dropna()
    if _is_numeric(ref):
        stat, p = stats.ks_2samp(ref, prod)
        test, value = "KS", float(stat)
        psi_value = psi(ref, prod)
        kind = "numeric"
    else:
        cats = sorted(set(ref.unique()) | set(prod.unique()), key=str)
        table = np.array([ref.value_counts().reindex(cats, fill_value=0),
                          prod.value_counts().reindex(cats, fill_value=0)])
        stat, p, _, _ = stats.chi2_contingency(table)
        test, value = "Chi²", float(stat)
        psi_value = psi(ref, prod, categorical=True)
        kind = "categorical"
    row = {
        "feature": name,
        "type": kind,
        "test": test,
        "statistic": round(value, 4),
        "p_value": float(p),
        "drifted": bool(p < P_VALUE_ALPHA),
        "psi": round(psi_value, 4),
        "severity": severity(psi_value),
        "critical": name in CRITICAL_FEATURES,
        "bootstrap": None,
    }
    row["disagree"] = tests_disagree(row)
    return row


def _bootstrap(ref: pd.Series, prod: pd.Series, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ref, prod = ref.dropna(), prod.dropna().to_numpy()
    categorical = not _is_numeric(ref)
    values = [psi(ref, rng.choice(prod, size=len(prod), replace=True), categorical=categorical)
              for _ in range(N_BOOTSTRAP)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    if hi < PSI_MODERATE:
        verdict, sev = "change too small to matter (PSI interval below 0.10)", "stable"
    elif lo >= PSI_MODERATE:
        verdict, sev = "drift confirmed (PSI interval above 0.10)", None
    else:
        verdict, sev = "borderline - treated as moderate (conservative)", "moderate"
    return {"ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "verdict": verdict, "resolved_severity": sev}


def detect(state: dict) -> dict:
    reference: pd.DataFrame = state["reference_df"]
    batch: pd.DataFrame = state["production_df"]
    bootstrap_for = state.get("bootstrap_features") or []
    rechecking = state.get("retry_target") == "detector"

    table = [_test_feature(reference[f], batch[f], f) for f in FEATURES]

    if rechecking and bootstrap_for:
        for row in table:
            if row["feature"] in bootstrap_for:
                b = _bootstrap(reference[row["feature"]], batch[row["feature"]])
                row["bootstrap"] = b
                if b["resolved_severity"]:
                    row["severity"] = b["resolved_severity"]
                row["disagree"] = False  # resolved by the bootstrap re-check

    # Prediction drift on champion risk scores
    champion = state["champion"]
    ref_scores = champion.predict_proba(state["reference_test_df"][FEATURES])[:, 1]
    prod_scores = champion.predict_proba(batch[FEATURES])[:, 1]
    pred_psi = psi(ref_scores, prod_scores)
    prediction_drift = {
        "psi": round(pred_psi, 4),
        "severity": severity(pred_psi),
        "ref_mean_score": float(ref_scores.mean()),
        "prod_mean_score": float(prod_scores.mean()),
        "ref_scores": ref_scores,
        "prod_scores": prod_scores,
    }

    performance = None
    if state.get("labels_available") and TARGET in batch.columns:
        current = evaluate(champion, batch)
        base = state["baseline_metrics"]
        performance = {
            "baseline_accuracy": base["accuracy"], "current_accuracy": current["accuracy"],
            "accuracy_drop": base["accuracy"] - current["accuracy"],
            "baseline_auc": base["auc"], "current_auc": current["auc"],
            "auc_drop": base["auc"] - current["auc"],
            "baseline_default_rate": base["default_rate"], "current_default_rate": current["default_rate"],
            "breach": (base["accuracy"] - current["accuracy"]) > MAX_ACCURACY_DROP,
        }

    major = [r["feature"] for r in table if r["severity"] == "major"]
    moderate = [r["feature"] for r in table if r["severity"] == "moderate"]
    if rechecking and bootstrap_for:
        action = "Bootstrap re-check"
        summary = "Re-checked " + ", ".join(f"`{f}`" for f in bootstrap_for) + " with a 300-sample bootstrap: " + "; ".join(
            f"`{r['feature']}` PSI 95% CI [{r['bootstrap']['ci_low']}, {r['bootstrap']['ci_high']}] → {r['bootstrap']['verdict']}"
            for r in table if r["bootstrap"]) + "."
    else:
        action = "Ran drift tests"
        summary = (f"Tested {len(table)} features. Major drift: {', '.join(major) or 'none'}. "
                   f"Moderate: {', '.join(moderate) or 'none'}. Prediction PSI {pred_psi:.3f} ({severity(pred_psi)}).")
        if performance:
            summary += f" Accuracy {performance['current_accuracy']:.1%} vs baseline {performance['baseline_accuracy']:.1%}."
    status = "warn" if major or moderate or prediction_drift["severity"] != "stable" else "ok"

    return {
        "drift_table": table,
        "prediction_drift": prediction_drift,
        "performance": performance,
        "retry_target": None if rechecking else state.get("retry_target"),
        "trace": log(state, "Drift Detector", action, summary, status,
                     {"major": major, "moderate": moderate, "prediction_psi": round(pred_psi, 4),
                      "accuracy_drop": round(performance["accuracy_drop"], 4) if performance else None}),
        "path": state.get("path", []) + ["detector"],
    }
