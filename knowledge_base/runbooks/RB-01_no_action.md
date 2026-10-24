---
id: RB-01
title: Runbook: No action
category: runbook
tags: no action, stable, monitoring, log
---

# RB-01 — Runbook: No action

When: all features and predictions are stable (PSI < 0.10), data quality is fine.

## Steps
1. Log the monitoring result; nobody is paged.
2. Keep the current champion in production.
3. Re-run the Watchdog on the next production batch as scheduled.
