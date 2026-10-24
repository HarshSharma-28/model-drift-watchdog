---
id: C-05
title: Concept drift vs data drift
category: concept
tags: concept drift, label, relationship change, retrain
---

# C-05 — Concept drift vs data drift

- **Data drift**: P(X) changes — the inputs look different.
- **Concept drift**: P(y | X) changes — the same inputs now lead to different outcomes.

In the Sudden-drift scenario both happen: more gig workers arrive (data drift) *and* gig workers from the new
channel default more than before (concept drift). Concept drift can only be confirmed once labels arrive, which is
why the Watchdog adds an accuracy check when labels are available and evaluates a retrained challenger on fresh
labelled data (C-08).

Without labels, data drift on critical features plus prediction drift (C-06) is the best early proxy.
