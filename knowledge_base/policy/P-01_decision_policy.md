---
id: P-01
title: Decision policy
category: policy
tags: decision, policy, no action, alert, retrain, fix pipeline, thresholds
---

# P-01 — Decision policy

The Decision Agent applies these rules in order (first match wins):

| Condition | Decision |
|---|---|
| Missing values > 10% on any feature, or schema mismatch | Fix pipeline |
| Major drift (PSI > 0.25) on a critical feature, prediction PSI > 0.25, or accuracy drop > 5 points | Retrain |
| Moderate drift (PSI 0.10–0.25) anywhere, major drift on a non-critical feature, or moderate prediction drift | Alert |
| Otherwise | No action |

Rationale:
- Data-quality problems come first: retraining on broken data would teach the model from corrupted inputs.
- Retrain is reserved for large, business-relevant change, because retraining has cost and risk.
- Alerts keep humans informed of slower trends without over-reacting.

Decisions are made by these rules, not by the LLM, so they are auditable and repeatable.
