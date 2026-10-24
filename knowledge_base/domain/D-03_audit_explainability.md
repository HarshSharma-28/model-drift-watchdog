---
id: D-03
title: Auditability and explainability
category: domain
tags: audit, explainability, regulation, compliance, transparency, trace
---

# D-03 — Auditability and explainability

Lenders are expected to be able to explain and audit automated credit decisions and the models behind them.
The Watchdog supports this by:
- Making every decision from a fixed, documented policy (P-01) rather than a free-form LLM answer.
- Recording which rule fired and the evidence behind it.
- Keeping a step-by-step agent trace, the Critic's checklist for every round, and a model registry with approvers.
- Citing the knowledge-base entries (runbooks, policy) that justify each action.
