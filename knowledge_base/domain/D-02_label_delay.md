---
id: D-02
title: Label delay
category: domain
tags: label delay, delayed outcomes, ground truth, early warning
---

# D-02 — Label delay

In lending, the true outcome (did the borrower default?) is known only months after the loan is issued.
So model accuracy cannot be measured in real time.

Consequences for monitoring:
- Input drift and prediction drift are the early-warning signals.
- Performance checks are added only when labels for a batch become available.
- The Watchdog runs correctly with labels switched off (same decisions in all four scenarios).
