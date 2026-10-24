---
id: C-01
title: Data drift
category: concept
tags: data drift, covariate shift, input distribution, monitoring
---

# C-01 — Data drift

Data drift (covariate shift) means the distribution of the model's **inputs** in production is different from the
distribution it was trained on. The model's code and weights do not change, so nothing crashes, but its predictions
are made for a population it has not seen.

Typical causes in lending:
- Economic change: inflation raises incomes and loan sizes.
- New acquisition channels: a partner brings applicants with a different profile (e.g. more gig workers).
- Product or policy change: new loan limits or eligibility rules.

Why it matters: a credit model that silently drifts approves risky loans or rejects good customers.
Because true outcomes (defaults) arrive months later, **input drift is the earliest warning signal**.

How the Watchdog detects it: per-feature statistical tests (KS for numeric, Chi-square for categorical) for
significance, and PSI for size. See C-02, C-03, C-04.
