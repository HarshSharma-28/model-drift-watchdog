---
id: RB-04
title: Runbook: Fix pipeline (data-quality incident)
category: runbook
tags: fix pipeline, missing values, schema, data engineering, incident, broken data
---

# RB-04 — Runbook: Fix pipeline (data-quality incident)

When: more than 10% missing values on any feature, or a schema mismatch.

## Steps
1. Do NOT retrain — the data is corrupted, not genuinely different.
2. Open an incident for the Data Engineering on-call with the affected columns and missing rates.
3. Check the upstream source (e.g. the credit bureau API) for failures or schema changes.
4. Restore the feed and back-fill the missing values where possible.
5. Re-run the Watchdog on the repaired batch before taking any model action.
