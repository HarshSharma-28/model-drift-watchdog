---
id: RB-02
title: Runbook: Alert (moderate drift)
category: runbook
tags: alert, moderate drift, watch, trend, ml team
---

# RB-02 — Runbook: Alert (moderate drift)

When: moderate drift on any feature, or major drift on a non-critical feature.

## Steps
1. Post the alert to the ML monitoring channel with the drifted features and their PSI.
2. Check whether the shift has a known business cause (inflation, a marketing campaign, a new region).
3. Increase monitoring frequency for the drifted features for the next few batches.
4. If PSI keeps rising past 0.25 on a critical feature, the next run will recommend Retrain.
