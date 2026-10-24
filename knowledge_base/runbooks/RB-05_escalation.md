---
id: RB-05
title: Runbook: Escalation to a human
category: runbook
tags: escalation, human review, insufficient evidence, critic
---

# RB-05 — Runbook: Escalation to a human

When: the Critic's checks still fail after 2 retries (for example the batch is too small even after widening the
window).

## Steps
1. Take no automatic action; keep the current champion.
2. Assign the run to an ML engineer for manual review of the evidence in the report.
3. Collect more data or fix the root cause the Critic reported.
4. Re-run the Watchdog once the issue is resolved.
