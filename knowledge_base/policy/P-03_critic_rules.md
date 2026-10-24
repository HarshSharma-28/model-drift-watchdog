---
id: P-03
title: Critic validation rules
category: policy
tags: critic, validation, retry, escalation, sample size, agreement
---

# P-03 — Critic validation rules

Before anything is acted on, the Critic runs five checks. The first failing check decides where work goes back.

| Check | If it fails |
|---|---|
| Sample size ≥ 500 rows | Profiler doubles the data window |
| KS/Chi-square and PSI agree on critical features | Detector runs a bootstrap re-check (C-07) |
| Never retrain on corrupted data | Decision agent revises |
| Decision matches the policy (re-derived independently) | Decision agent revises |
| Decision is backed by recorded evidence | Decision agent revises |

At most 2 retries. If checks still fail, the run is escalated to a human (RB-05) and no automatic action is taken.
