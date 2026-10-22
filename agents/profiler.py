"""Data Profiler Agent (role: Research).

Reads the monitored window of production data and compares it with the
reference data: schema, missing-value rates, sample size and summary
statistics. Sets `data_ok`, which decides whether the Planner must re-plan.

When the Critic asks for a larger window (sample too small), the Profiler
reads further back into recent production data (window doubles each time).
"""

from __future__ import annotations

import pandas as pd

from agents.common import fmt_pct, log
from policy import MAX_MISSING_RATE, MIN_SAMPLE_SIZE, TARGET


def profile(state: dict) -> dict:
    reference: pd.DataFrame = state["reference_df"]
    pool: pd.DataFrame = state["production_pool"]

    retrying = state.get("retry_target") == "profiler"
    window = state.get("window_size") or state.get("batch_size", 1000)
    if retrying:
        window = min(len(pool), window * 2)
    window = min(window, len(pool))
    batch = pool.head(window).reset_index(drop=True)

    expected_cols = [c for c in reference.columns if c != TARGET]
    missing_cols = [c for c in expected_cols if c not in batch.columns]
    extra_cols = [c for c in batch.columns if c not in reference.columns]
    type_mismatch = [
        c for c in expected_cols
        if c in batch.columns and pd.api.types.is_numeric_dtype(reference[c]) != pd.api.types.is_numeric_dtype(batch[c])
    ]

    missing_rates = {c: float(batch[c].isna().mean()) for c in expected_cols if c in batch.columns}
    high_missing = {c: r for c, r in missing_rates.items() if r > MAX_MISSING_RATE}

    issues = []
    if missing_cols:
        issues.append(f"missing columns: {', '.join(missing_cols)}")
    if type_mismatch:
        issues.append(f"type mismatch: {', '.join(type_mismatch)}")
    for c, r in high_missing.items():
        issues.append(f"`{c}` {fmt_pct(r)} missing (limit {fmt_pct(MAX_MISSING_RATE)})")

    stats = {}
    for c in expected_cols:
        if c not in batch.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference[c]) and reference[c].nunique() > 10:
            stats[c] = {
                "ref_mean": float(reference[c].mean()), "prod_mean": float(batch[c].mean()),
                "ref_std": float(reference[c].std()), "prod_std": float(batch[c].std()),
            }
        else:
            stats[c] = {
                "ref_share": reference[c].value_counts(normalize=True).round(3).to_dict(),
                "prod_share": batch[c].value_counts(normalize=True).round(3).to_dict(),
            }

    data_ok = not issues
    report = {
        "rows": int(len(batch)),
        "window_size": int(window),
        "min_sample_size": MIN_SAMPLE_SIZE,
        "sample_ok": len(batch) >= MIN_SAMPLE_SIZE,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "type_mismatch": type_mismatch,
        "missing_rates": missing_rates,
        "issues": issues,
        "data_ok": data_ok,
        "stats": stats,
    }

    if retrying:
        action = "Widened data window"
        summary = f"Critic asked for more data: window widened to the latest {window:,} rows. "
    else:
        action = "Profiled data quality"
        summary = f"Profiled the latest {window:,} production rows against {len(reference):,} reference rows. "
    summary += ("Schema and missing values look healthy." if data_ok else "Problems found: " + "; ".join(issues) + ".")
    if len(batch) < MIN_SAMPLE_SIZE:
        summary += f" Sample is small ({len(batch)} < {MIN_SAMPLE_SIZE})."

    return {
        "window_size": window,
        "production_df": batch,
        "quality_report": report,
        "retry_target": None if retrying else state.get("retry_target"),
        "trace": log(state, "Data Profiler", action, summary, "ok" if data_ok else "warn",
                     {"rows": len(batch), "max_missing_rate": max(missing_rates.values()) if missing_rates else 0,
                      "issues": issues}),
        "path": state.get("path", []) + ["profiler"],
    }
