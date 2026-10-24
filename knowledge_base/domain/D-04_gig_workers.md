---
id: D-04
title: Gig-worker applicants
category: domain
tags: gig workers, thin file, new to credit, income volatility, segment
---

# D-04 — Gig-worker applicants

Gig workers (delivery, ride-hailing, freelance) often have irregular income and thin credit files.
For a model trained mostly on salaried applicants this is a different population:
- Credit scores are lower or less informative.
- Loan-to-income ratios matter more because income is volatile.

A sudden influx of such applicants (e.g. via a new partner channel) is a classic cause of both data drift and
concept drift (C-05), and usually justifies a Retrain with recent labelled data (RB-03).
