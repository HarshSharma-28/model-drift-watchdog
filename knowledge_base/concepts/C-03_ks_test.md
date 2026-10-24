---
id: C-03
title: Kolmogorov–Smirnov (KS) test
category: concept
tags: ks test, kolmogorov smirnov, numeric, p-value, significance
---

# C-03 — Kolmogorov–Smirnov (KS) test

The two-sample KS test checks whether two samples of a **numeric** feature come from the same distribution.
The statistic D is the largest gap between the two empirical cumulative distribution functions.

- p-value < 0.05 → the difference is statistically significant (drifted).
- It makes no assumption about the shape of the distribution (non-parametric).

Caveat: with 1,000+ rows, even a very small shift can be "significant". That is why the Watchdog pairs KS with PSI.
If KS is significant but PSI is below 0.10 (or the reverse) on a critical feature, the Critic asks for a bootstrap
re-check (C-07).

Used by the Planner for: age, monthly_income, loan_amount, credit_score.
