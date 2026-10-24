---
id: C-07
title: Bootstrap re-check
category: concept
tags: bootstrap, resampling, confidence interval, disagreement, critic
---

# C-07 — Bootstrap re-check

When KS/Chi-square and PSI disagree on a critical feature, the Drift Detector re-checks it with a bootstrap:
the production batch is resampled with replacement 300 times, PSI is recomputed each time, and the 95% interval
is taken.

| Bootstrap result | Interpretation |
|---|---|
| Upper bound < 0.10 | Change too small to matter → treated as stable |
| Lower bound ≥ 0.10 | Drift confirmed |
| Interval straddles 0.10 | Borderline → treated as moderate (conservative) |

This settles statistical noise vs real change without a human, and it is one of the Critic's retry paths (P-03).
