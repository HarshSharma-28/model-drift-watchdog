---
id: C-06
title: Prediction drift
category: concept
tags: prediction drift, score distribution, risk scores, psi
---

# C-06 — Prediction drift

Prediction drift compares the distribution of the model's **output scores** (predicted default probability) on the
reference test set vs the production batch, using PSI.

It is useful because it summarises the effect of all input changes on what the model actually does. A prediction
PSI above 0.25 means the model is now scoring the population very differently and triggers Retrain under the
policy (P-01), even if no single feature looks alarming.
