---
id: RB-03
title: Runbook: Retrain
category: runbook
tags: retrain, major drift, challenger, approval, deploy, credit risk lead
---

# RB-03 — Runbook: Retrain

When: major drift on a critical feature, prediction PSI > 0.25, or accuracy dropped more than 5 points.

## Steps
1. Notify the ML team and the Credit Risk lead; approval will be needed.
2. Train a challenger on reference data plus recent labelled data (recent rows weighted 3×).
3. Compare champion vs challenger on fresh, unseen data (C-08).
4. If the AUC gain is at least 0.01, request human approval; otherwise keep the champion and keep monitoring.
5. On approval, deploy the challenger and record it in the model registry (P-04).
6. Consider tightening credit rules for the drifted segment until the new model is live.
