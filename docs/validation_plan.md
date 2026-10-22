# Validation and Results

The project uses synthetic data with **known, injected drift**, so every agent output can be checked
against ground truth. Reproduce everything below with:

```bash
python validation.py      # results tables
python -m pytest -q       # 25 automated tests
```

or press **Run validation suite** in the app's Validation tab.

## 1. Scenario ground truth — decision accuracy 8 / 8

| Scenario | Injected change | Labels | Expected | Agent decision | Match | Critic retries |
|---|---|---|---|---|---|---|
| Normal week | None | yes | No action | No action | ✅ | 0 |
| Gradual drift | Incomes +20% (loan amounts follow), more tier-2/3 applicants | yes | Alert | Alert | ✅ | 1 |
| Sudden drift | Gig-worker influx, lower credit scores, gig workers riskier | yes | Retrain | Retrain | ✅ | 0 |
| Pipeline bug | 40% of `credit_score` missing | yes | Fix pipeline | Fix pipeline | ✅ | 0 |
| Normal week | None | no | No action | No action | ✅ | 0 |
| Gradual drift | as above | no | Alert | Alert | ✅ | 1 |
| Sudden drift | as above | no | Retrain | Retrain | ✅ | 0 |
| Pipeline bug | as above | no | Fix pipeline | Fix pipeline | ✅ | 0 |

"Labels = no" simulates label delay: the decision is made from input and prediction drift alone.

## 2. Feature-level detection — precision 100%, recall 100%

A feature counts as *flagged* if its KS / Chi-square test is significant (p < 0.05) or its PSI ≥ 0.10.

| Scenario | Injected features | Flagged features | Key numbers |
|---|---|---|---|
| Normal week | — | — | max PSI 0.005, prediction PSI 0.009 |
| Gradual drift | monthly_income, loan_amount, city_tier | same | income PSI 0.136 (moderate); loan_amount PSI 0.085, significant → bootstrap re-check |
| Sudden drift | credit_score, employment_type, monthly_income, loan_amount | same | credit_score PSI 0.887, employment_type PSI 0.637 (major); prediction PSI 0.863 |
| Pipeline bug | credit_score (missing values) | same | 38.4% missing in the batch (limit 10%) |

## 3. Critic checks (live in the UI)

| Check | If it fails |
|---|---|
| Sample size ≥ 500 | Profiler widens the window (doubles) |
| KS/Chi² and PSI agree on critical features | Detector re-checks with a 300-sample bootstrap of PSI |
| Never retrain on corrupted data | Decision agent revises |
| Decision follows the policy (re-derived independently) | Decision agent revises |
| Decision is backed by evidence | Decision agent revises |

Maximum 2 retries, then escalate to a human.

## 4. Robustness tests — 7 / 7 passed

| Test | What happened | Passed |
|---|---|---|
| Faulty first decision — Normal week | Naive 'Retrain' → Critic retry (decision) → No action | ✅ |
| Faulty first decision — Gradual drift | Naive 'Retrain' → Critic retry (detector, then decision) → Alert | ✅ |
| Faulty first decision — Sudden drift | Naive 'No action' → Critic retry (decision) → Retrain | ✅ |
| Faulty first decision — Pipeline bug | Naive 'Retrain' on broken data → Critic blocks it → Fix pipeline | ✅ |
| Small batch (300 rows) | Window widened to 600 rows → No action | ✅ |
| Tiny batch (100 rows) | 400 rows after 2 retries → escalated to human, no retrain | ✅ |
| LangGraph vs fallback runner | Identical decision and path | ✅ |

## 5. Model check — champion vs challenger (Sudden drift)

| Model | AUC (2,000 fresh rows) | Accuracy |
|---|---|---|
| Champion (deployed) | 0.742 | 0.719 |
| Challenger (retrained on reference + 1,000 recent labelled rows, recent ×3) | **0.768** | 0.726 |
| Gain | **+0.026** (bar: ≥ 0.01) → eligible | |

On the monitored batch the champion's accuracy had fallen from 83.5% (baseline) to 71.6%.

## 6. Human check

No model is deployed without an explicit **Approve & deploy** click. Approved models are recorded in the
model registry (v1 → v2), and later monitoring runs use the new champion.

## Success criteria — all met

- [x] 4 / 4 scenarios produce the expected decision (8 / 8 including label-delay runs)
- [x] The Pipeline-bug scenario is **not** sent for retraining
- [x] In the Sudden-drift scenario, the retrained model beats the champion on fresh data
- [x] The Critic catches every injected faulty decision and escalates when evidence stays insufficient
