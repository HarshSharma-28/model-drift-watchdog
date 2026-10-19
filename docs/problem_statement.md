# Problem Statement and Domain

**Project:** Model Drift Watchdog
**Team:** Harsh Sharma (24070126067), Divyansh Agarwal (24070126056)
**Domain:** MLOps — credit risk model at a digital lending company

## Problem statement

Machine learning models are trained on past data, but the real world keeps changing. After deployment, the data a model receives slowly or suddenly stops looking like the data it was trained on. This is **data drift**.

Examples in lending:
- Applicant incomes rise with inflation.
- A new customer segment (gig workers) starts applying in large numbers.
- An upstream data pipeline breaks and sends missing credit scores.

The model keeps running without errors, but its decisions get worse: risky loans are approved and good customers are rejected. Because loan defaults are only known months later, the damage is usually discovered too late.

## Why this domain

| Reason | Explanation |
|---|---|
| High financial stakes | Every wrong prediction is a bad loan or a lost customer. |
| Label delay | True outcomes arrive months later, so real-time accuracy monitoring is impossible. Input-data monitoring is the only early warning. |
| Regulatory risk | Lenders must explain and audit automated credit decisions. |
| Real industry pain | Monitoring and incident investigation consume a large share of ML teams' time. |

## Existing business challenge

- Drift checks are done **manually**, monthly or ad-hoc, in notebooks.
- Problems are noticed **after** losses appear.
- Engineers struggle to tell **real drift** apart from a **broken pipeline**, and sometimes retrain on bad data.
- The decision to retrain is a **judgement call** with no consistent policy.
- There is **no standard report** or audit trail.

## Proposed solution

A team of cooperating AI agents that, for every new batch of production data:

1. **Plans** which checks to run based on the data (feature types, whether labels exist).
2. **Profiles** the data for quality problems and re-plans if the data itself is broken.
3. **Detects** drift with statistical tests (KS, Chi-square, PSI) and prediction drift.
4. **Decides** on an action using a fixed, auditable policy: No action, Alert, Retrain or Fix pipeline.
5. **Validates** the evidence and decision with a Critic agent that can trigger retries or escalate.
6. **Acts** by writing a report and alert, and — when needed — training a challenger model and comparing it with the current champion.
7. **Asks a human** to approve before any new model is deployed.

## Expected impact

- Faster detection: every batch instead of once a month.
- Consistent, policy-based decisions.
- Fewer false alarms and no retraining on broken data.
- A full audit trail of every check and decision.
