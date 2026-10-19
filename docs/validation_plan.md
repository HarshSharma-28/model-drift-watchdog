# Validation Plan

The project uses synthetic data with **known, injected drift**, so every output can be checked against ground truth.

## 1. Scenario ground truth

| Scenario | Injected change | Expected decision | Actual decision | Match |
|---|---|---|---|---|
| Normal week | None | No action | _to be filled after run_ | |
| Gradual drift | Incomes +~20%, city tier shift | Alert | | |
| Sudden drift | Gig-worker influx, lower credit scores | Retrain | | |
| Pipeline bug | 40% `credit_score` missing | Fix pipeline | | |

## 2. Feature-level detection

For each scenario, compare the features that were actually drifted with the features the Detector flagged, and report precision and recall of drift detection.

## 3. Critic checks (live in the UI)

- Sample size ≥ 500
- KS and PSI agree on critical features
- Decision consistent with policy
- Retries ≤ 2, otherwise escalation

## 4. Model check

In the Retrain scenario, the challenger model must beat the champion's AUC by at least 0.01 on a fresh hold-out set before it can be proposed for deployment.

## 5. Human check

No model is deployed without an explicit Approve click in the UI.

## Success criteria

- 4 / 4 scenarios produce the expected decision.
- The Pipeline-bug scenario is **not** sent for retraining.
- In the Sudden-drift scenario, the retrained model recovers lost AUC.
